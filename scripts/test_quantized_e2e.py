"""
TurboGene: End-to-end quantized model test.

Tests:
1. Log-likelihood equivalence (orig vs TurboGene)
2. VRAM profiling at batch=1,4,8,16 (standard + memory-efficient)
3. "batch=8 was OOM, now it fits" proof
"""

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024


def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_batch(bs, seq_len, vocab_size, device="cuda"):
    gen = torch.Generator(device="cpu").manual_seed(42)
    gids = torch.randint(7, vocab_size, (bs, seq_len), generator=gen).to(device)
    counts = torch.randint(1, 30, (bs, seq_len), generator=gen).float().to(device)
    pad_start = seq_len - seq_len // 10
    gids[:, pad_start:] = 5
    counts[:, pad_start:] = 0
    return gids, counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda"
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_total:.0f} MB)")

    # Load and convert
    print("\n1. Loading original TranscriptFormer...")
    from scripts.test_sdpa_e2e import load_original_model
    orig, vocab = load_original_model(args.checkpoint_dir)
    orig = orig.to(device, torch.float32).eval()
    vocab_size = len(vocab)
    mc = orig.model_config
    seq_len = mc.seq_len + mc.aux_len

    print("2. Converting to SDPA...")
    from turbogene.sdpa_model import SDPATranscriptformer
    sdpa = SDPATranscriptformer(orig).to(device, torch.float32).eval()

    print("3. Calibrating quantizer...")
    from turbogene.quantizer import TurboGeneQuantizer

    num_heads = mc.num_heads
    head_dim = mc.embed_dim // num_heads
    quantizer = TurboGeneQuantizer(mc.num_layers, num_heads, head_dim, n_bits=3).to(device)

    k_acts, v_acts = [], []
    def make_hook(li):
        def hook(mod, args):
            inp = args[0]
            B = inp.size(0)
            k = mod.linears[1](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2)
            v = mod.linears[2](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2)
            k_acts.append((li, k.detach()))
            v_acts.append((li, v.detach()))
        return hook

    hooks = []
    for i, layer in enumerate(sdpa.transformer_encoder.encoder_layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(make_hook(i)))
    cal_g, cal_c = make_batch(2, seq_len, vocab_size)
    with torch.no_grad():
        _ = sdpa(gene_token_indices=cal_g, gene_counts=cal_c)
    for h in hooks:
        h.remove()
    del cal_g, cal_c
    quantizer.calibrate(k_acts, v_acts)
    del k_acts, v_acts
    print(f"   Calibrated: {mc.num_layers}L x {num_heads}H, 3-bit + QJL")

    print("4. Creating quantized model...")
    from turbogene.quantized_model import QuantizedTranscriptformer
    qmodel = QuantizedTranscriptformer(sdpa, quantizer).to(device, torch.float32).eval()

    # Log-likelihood comparison
    print("\n" + "=" * 60)
    print("5. Log-Likelihood (batch=1, standard forward = no quant error)")
    print("=" * 60)

    gids, counts = make_batch(1, seq_len, vocab_size)
    from transcriptformer.data.dataclasses import BatchData
    batch = BatchData(gene_token_indices=gids, gene_counts=counts, aux_token_indices=None)

    with torch.no_grad():
        out_orig = orig(batch, embed=True)
        out_q = qmodel(gene_token_indices=gids, gene_counts=counts, embed=True)

    mu_cos = F.cosine_similarity(
        out_orig["mu"].reshape(1, -1), out_q["mu"].reshape(1, -1), dim=-1
    ).item()
    llh_o = orig.criterion(mu=out_orig["mu"], input_counts=out_orig["input_counts"], mask=out_orig["mask"], eval_mode=True)
    llh_q = qmodel.criterion(mu=out_q["mu"], input_counts=out_q["input_counts"], mask=out_q["mask"], eval_mode=True)
    print(f"   mu cos_sim:    {mu_cos:.10f}")
    print(f"   LLH orig:      {llh_o.mean().item():.4f}")
    print(f"   LLH TurboGene: {llh_q.mean().item():.4f}")
    print(f"   LLH diff:      {abs(llh_o.mean() - llh_q.mean()).item():.2e}")
    if "embeddings" in out_orig and "embeddings" in out_q:
        ec = F.cosine_similarity(out_orig["embeddings"].reshape(1,-1), out_q["embeddings"].reshape(1,-1), dim=-1).item()
        print(f"   Emb cos_sim:   {ec:.10f}")
    del out_orig, out_q, batch, gids, counts

    # VRAM Profiling
    print("\n" + "=" * 60)
    print("6. VRAM Profiling")
    print("=" * 60)

    del orig, sdpa
    reset()
    qmodel = qmodel.half()
    reset()
    wm = get_mb()
    print(f"   Weights (FP16): {wm:.0f} MB")

    print("\n   --- Standard Forward ---")
    for bs in [1, 2, 4, 6, 8, 10, 12, 16]:
        reset()
        try:
            gids, counts = make_batch(bs, seq_len, vocab_size)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                result = qmodel(gids, counts)
            p = peak_mb()
            print(f"   batch={bs:2d}: peak={p:.0f} MB ({p/gpu_total*100:.1f}%)")
            del result, gids, counts
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   batch={bs:2d}: OOM!")
                torch.cuda.empty_cache()
            else:
                raise

    print("\n   --- Memory-Efficient Forward (chunked attention) ---")
    for bs in [1, 2, 4, 6, 8, 10, 12, 16]:
        reset()
        try:
            gids, counts = make_batch(bs, seq_len, vocab_size)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                result = qmodel.forward_memory_efficient(gids, counts)
            p = peak_mb()
            print(f"   batch={bs:2d}: peak={p:.0f} MB ({p/gpu_total*100:.1f}%)")
            del result, gids, counts
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   batch={bs:2d}: OOM!")
                torch.cuda.empty_cache()
            else:
                raise

    # Quality check
    print("\n" + "=" * 60)
    print("7. Memory-Efficient Quality Check (batch=1, FP32)")
    print("=" * 60)
    reset()
    qf32 = qmodel.float()
    gids, counts = make_batch(1, seq_len, vocab_size)
    with torch.no_grad():
        o_std = qf32(gids, counts)
        o_eff = qf32.forward_memory_efficient(gids, counts)
    mc2 = F.cosine_similarity(o_std["mu"].reshape(1,-1), o_eff["mu"].reshape(1,-1), dim=-1).item()
    print(f"   Standard vs chunked mu cos: {mc2:.10f}")
    print(f"   {'PASS' if mc2 > 0.999 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
