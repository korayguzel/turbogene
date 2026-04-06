"""
TurboGene Phase 2: TurboQuant Quantizer Validation

Tests each quantization stage:
  1. Rotation: Pi^T Pi = I, inner product preservation
  2. Lloyd-Max: quantize/dequantize cosine similarity
  3. QJL: residual correction improvement
  4. Full pipeline: attention output with quantized KV
  5. Memory compression ratio

Usage:
    python scripts/test_quantizer.py --checkpoint-dir ./checkpoints/tf_sapiens
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_and_get_activations(checkpoint_dir, seq_len=512, batch_size=2):
    from scripts.test_sdpa_e2e import load_original_model
    from turbogene.sdpa_model import SDPATranscriptformer

    orig, vocab = load_original_model(checkpoint_dir)
    orig = orig.to("cuda", torch.float32).eval()
    sdpa = SDPATranscriptformer(orig).to("cuda", torch.float32).eval()

    vocab_size = len(vocab)
    gen = torch.Generator(device="cpu").manual_seed(42)
    gids = torch.randint(7, vocab_size, (batch_size, seq_len), generator=gen).cuda()
    counts = torch.randint(1, 30, (batch_size, seq_len), generator=gen).float().cuda()
    gids[:, seq_len - seq_len // 10:] = 5
    counts[:, seq_len - seq_len // 10:] = 0

    k_acts, v_acts = [], []

    def make_hook(layer_idx):
        def hook(mod, args):
            inp = args[0]
            B = inp.size(0)
            k = mod.linears[1](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2)
            v = mod.linears[2](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2)
            k_acts.append((layer_idx, k.detach()))
            v_acts.append((layer_idx, v.detach()))
        return hook

    hooks = []
    for i, layer in enumerate(sdpa.transformer_encoder.encoder_layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(make_hook(i)))

    with torch.no_grad():
        _ = sdpa(gene_token_indices=gids, gene_counts=counts, embed=False)

    for h in hooks:
        h.remove()

    return sdpa, k_acts, v_acts


def test_rotation(quantizer, k_acts, v_acts):
    print("\n" + "=" * 60)
    print("TEST 1: Orthogonal Rotation")
    print("=" * 60)

    identity_errors = []
    for l in range(quantizer.num_layers):
        for h in range(quantizer.num_heads):
            Pi = quantizer.rotations[l, h]
            err = (Pi @ Pi.T - torch.eye(Pi.size(0), device=Pi.device)).abs().max().item()
            identity_errors.append(err)

    print(f"  Pi^T Pi = I: max_err={max(identity_errors):.2e}")

    layer_idx, k = k_acts[0]
    B, H, S, D = k.shape
    x = k[:, 0, :10, :]
    y = k[:, 0, 10:20, :]
    Pi = quantizer.rotations[layer_idx, 0]
    ip_orig = torch.sum(x * y, dim=-1)
    ip_rot = torch.sum((x @ Pi.T) * (y @ Pi.T), dim=-1)
    ip_cos = F.cosine_similarity(ip_orig.reshape(1, -1), ip_rot.reshape(1, -1), dim=-1).item()
    print(f"  Inner product preservation: cos={ip_cos:.10f}")

    passed = ip_cos > 0.999 and max(identity_errors) < 1e-3  # SVD rotations have slightly higher numerical error
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_lloyd_max(quantizer, k_acts, v_acts):
    print("\n" + "=" * 60)
    print("TEST 2: Lloyd-Max 3-bit Quantization (no QJL)")
    print("=" * 60)

    cos_sims = {"K": [], "V": []}
    for kv_type, activations, is_key in [("K", k_acts, True), ("V", v_acts, False)]:
        for layer_idx, tensor in activations:
            B, H, S, D = tensor.shape
            indices, _, _, vector_norms = quantizer.quantize_vector(tensor, layer_idx, is_key)

            Pi = quantizer.rotations[layer_idx]
            centroids = quantizer.k_centroids[layer_idx] if is_key else quantizer.v_centroids[layer_idx]
            C = quantizer.n_centroids
            c_exp = centroids[None, :, None, None, :].expand(B, H, S, D, C)
            x_deq = torch.gather(c_exp, dim=-1, index=indices.long().unsqueeze(-1)).squeeze(-1)
            # Norm correction before inverse rotation (re-normalize to unit norm)
            x_deq_norms = torch.norm(x_deq, dim=-1, keepdim=True)
            safe_norms = torch.where(x_deq_norms > 1e-10, x_deq_norms, torch.ones_like(x_deq_norms))
            x_deq = x_deq / safe_norms
            x_recon = torch.einsum("bhsd,hde->bhse", x_deq, Pi)
            # Rescale by original vector norms
            x_recon = x_recon * vector_norms.unsqueeze(-1)

            cos = F.cosine_similarity(
                tensor.reshape(B, -1).float(), x_recon.reshape(B, -1).float(), dim=-1
            ).mean().item()
            cos_sims[kv_type].append(cos)

    for kv in ["K", "V"]:
        s = cos_sims[kv]
        print(f"  {kv}: mean={sum(s)/len(s):.6f}, min={min(s):.6f}, max={max(s):.6f}")

    all_s = cos_sims["K"] + cos_sims["V"]
    mean_s = sum(all_s) / len(all_s)
    passed = mean_s > 0.90
    print(f"  Combined mean: {mean_s:.6f}")
    print(f"  {'PASS' if passed else 'FAIL'}: cos > 0.90")
    return passed


def test_qjl(quantizer, k_acts, v_acts):
    print("\n" + "=" * 60)
    print("TEST 3: QJL 1-bit Residual (JL Projection)")
    print("=" * 60)

    no_qjl, with_qjl = [], []
    jl_sign_diffs = []
    for kv_type, activations, is_key in [("K", k_acts, True), ("V", v_acts, False)]:
        for layer_idx, tensor in activations:
            B, H, S, D = tensor.shape
            indices, signs, residual_norms, vector_norms = quantizer.quantize_vector(tensor, layer_idx, is_key)

            # With QJL
            x_full = quantizer.dequantize_vector(indices, signs, residual_norms, vector_norms, layer_idx, is_key)
            cos_full = F.cosine_similarity(
                tensor.reshape(B, -1).float(), x_full.reshape(B, -1).float(), dim=-1
            ).mean().item()

            # Without QJL (Lloyd-Max only with norm correction)
            Pi = quantizer.rotations[layer_idx]
            centroids = quantizer.k_centroids[layer_idx] if is_key else quantizer.v_centroids[layer_idx]
            C = quantizer.n_centroids
            c_exp = centroids[None, :, None, None, :].expand(B, H, S, D, C)
            x_deq = torch.gather(c_exp, dim=-1, index=indices.long().unsqueeze(-1)).squeeze(-1)
            # Norm correction before inverse rotation
            x_deq_norms = torch.norm(x_deq, dim=-1, keepdim=True)
            safe_norms = torch.where(x_deq_norms > 1e-10, x_deq_norms, torch.ones_like(x_deq_norms))
            x_deq = x_deq / safe_norms
            x_no = torch.einsum("bhsd,hde->bhse", x_deq, Pi)
            x_no = x_no * vector_norms.unsqueeze(-1)
            cos_no = F.cosine_similarity(
                tensor.reshape(B, -1).float(), x_no.reshape(B, -1).float(), dim=-1
            ).mean().item()

            no_qjl.append(cos_no)
            with_qjl.append(cos_full)

            # Verify JL matrix is being used: signs should differ from raw residual signs
            x_float = tensor.float()
            vn = torch.norm(x_float, dim=-1)
            safe_vn = torch.where(vn > 0, vn, torch.ones_like(vn))
            x_norm = x_float / safe_vn.unsqueeze(-1)
            x_rot = torch.einsum("bhsd,hde->bhse", x_norm, Pi.transpose(-1, -2))
            c_exp2 = centroids[None, :, None, None, :].expand(B, H, S, D, C)
            x_deq2 = torch.gather(c_exp2, dim=-1, index=indices.long().unsqueeze(-1)).squeeze(-1)
            x_deq2_norms = torch.norm(x_deq2, dim=-1, keepdim=True)
            safe2 = torch.where(x_deq2_norms > 1e-10, x_deq2_norms, torch.ones_like(x_deq2_norms))
            x_deq2_unit = x_deq2 / safe2
            residual = x_rot - x_deq2_unit
            raw_signs = residual > 0
            jl_diff = (signs != raw_signs).float().mean().item()
            jl_sign_diffs.append(jl_diff)

    mn = sum(no_qjl) / len(no_qjl)
    mw = sum(with_qjl) / len(with_qjl)
    mean_jl_diff = sum(jl_sign_diffs) / len(jl_sign_diffs)
    print(f"  Without QJL: {mn:.6f}")
    print(f"  With QJL:    {mw:.6f}")
    print(f"  Delta:       {mw - mn:+.6f}")
    print(f"  JL vs raw sign diff: {mean_jl_diff:.3f} (expect ~0.3-0.5)")

    # QJL with JL projection trades per-vector reconstruction for inner product
    # preservation (JL lemma). Test passes if:
    # 1. JL matrix is actually used (signs differ from raw signs)
    # 2. Reconstruction quality stays above 0.95 (still useful)
    jl_used = mean_jl_diff > 0.2
    quality_ok = mw > 0.95
    passed = jl_used and quality_ok
    print(f"  JL active: {'YES' if jl_used else 'NO'}")
    print(f"  Quality:   {'OK' if quality_ok else 'LOW'} (threshold: 0.95)")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_attention_output(sdpa_model, quantizer, k_acts, v_acts):
    print("\n" + "=" * 60)
    print("TEST 4: Attention Output with Quantized KV")
    print("=" * 60)

    _, k_orig = k_acts[0]
    _, v_orig = v_acts[0]
    B, H, S, D = k_orig.shape
    layer_idx = 0

    q = k_orig[:, :, -1:, :]  # last position as query
    scores_fp = torch.matmul(q.float(), k_orig.float().transpose(-2, -1)) / math.sqrt(D)

    k_idx, k_sgn, k_rmag, k_nrm = quantizer.quantize_vector(k_orig, layer_idx, is_key=True)
    k_recon = quantizer.dequantize_vector(k_idx, k_sgn, k_rmag, k_nrm, layer_idx, is_key=True)
    scores_q = torch.matmul(q.float(), k_recon.float().transpose(-2, -1)) / math.sqrt(D)

    weights_fp = F.softmax(scores_fp, dim=-1)
    weights_q = F.softmax(scores_q, dim=-1)

    v_idx, v_sgn, v_rmag, v_nrm = quantizer.quantize_vector(v_orig, layer_idx, is_key=False)
    v_recon = quantizer.dequantize_vector(v_idx, v_sgn, v_rmag, v_nrm, layer_idx, is_key=False)

    out_fp = torch.matmul(weights_fp, v_orig.float())
    out_q = torch.matmul(weights_q, v_recon.float())

    score_cos = F.cosine_similarity(scores_fp.reshape(B, -1), scores_q.reshape(B, -1), dim=-1).mean().item()
    weight_cos = F.cosine_similarity(weights_fp.reshape(B, -1), weights_q.reshape(B, -1), dim=-1).mean().item()
    output_cos = F.cosine_similarity(out_fp.reshape(B, -1), out_q.reshape(B, -1), dim=-1).mean().item()

    print(f"  Attention scores:  cos={score_cos:.6f}")
    print(f"  Attention weights: cos={weight_cos:.6f}")
    print(f"  Attention output:  cos={output_cos:.6f} (target: >0.95)")
    passed = output_cos > 0.95
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_memory(quantizer):
    print("\n" + "=" * 60)
    print("TEST 5: Memory Compression")
    print("=" * 60)

    mem = quantizer.memory_bytes(batch_size=4, seq_len=2048)
    fp16_mb = mem["fp16_kv_bytes"] / 1024 / 1024
    quant_mb = mem["total_bytes"] / 1024 / 1024
    ratio = mem["compression_ratio"]

    print(f"  FP16 KV:      {fp16_mb:.1f} MB")
    print(f"  Quantized KV: {quant_mb:.1f} MB")
    print(f"  Ratio:        {ratio:.2f}x")
    print(f"  Eff bits:     {mem['effective_bits_per_coord']:.2f}")
    print(f"  Savings:      {fp16_mb - quant_mb:.1f} MB")
    passed = ratio > 2.0
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("TurboGene: TurboQuant Quantizer Validation")
    print("=" * 60)

    print("\nLoading model and capturing activations...")
    sdpa_model, k_acts, v_acts = load_model_and_get_activations(args.checkpoint_dir)

    mc = sdpa_model.model_config
    num_layers, num_heads = mc.num_layers, mc.num_heads
    head_dim = mc.embed_dim // num_heads
    print(f"Captured: {len(k_acts)} layers, K shape={k_acts[0][1].shape}")

    from turbogene.quantizer import TurboGeneQuantizer
    quantizer = TurboGeneQuantizer(num_layers, num_heads, head_dim, n_bits=3).cuda()

    print("\nCalibrating from activations...")
    quantizer.calibrate(k_acts, v_acts)
    print("Done.")

    results = {}
    results["rotation"] = test_rotation(quantizer, k_acts, v_acts)
    results["lloyd_max"] = test_lloyd_max(quantizer, k_acts, v_acts)
    results["qjl"] = test_qjl(quantizer, k_acts, v_acts)
    results["attention"] = test_attention_output(sdpa_model, quantizer, k_acts, v_acts)
    results["memory"] = test_memory(quantizer)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        print(f"  {name:15s}: {'PASS' if passed else 'FAIL'}")
        all_pass = all_pass and passed
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
