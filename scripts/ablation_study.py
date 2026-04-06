"""
TurboGene Ablation Study and KIVI Baseline Comparison

Ablation:
  a) Rotation only  b) +Lloyd-Max  c) +QJL  d) Random vs SVD  e) Shared vs Dual LUT

Baseline:
  FP16, KIVI 2-bit, TurboGene 3-bit
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_activations(checkpoint_dir, seq_len=512, bs=2):
    from scripts.test_sdpa_e2e import load_original_model
    from turbogene.sdpa_model import SDPATranscriptformer

    orig, vocab = load_original_model(checkpoint_dir)
    orig = orig.to("cuda", torch.float32).eval()
    sdpa = SDPATranscriptformer(orig).to("cuda", torch.float32).eval()
    mc = orig.model_config

    k_acts, v_acts = [], []
    def mh(li):
        def hook(mod, args):
            inp = args[0]; B = inp.size(0)
            k_acts.append((li, mod.linears[1](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
            v_acts.append((li, mod.linears[2](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
        return hook
    hooks = [l.self_attn.register_forward_pre_hook(mh(i)) for i,l in enumerate(sdpa.transformer_encoder.encoder_layers)]
    sl = mc.seq_len + mc.aux_len
    gen = torch.Generator(device="cpu").manual_seed(42)
    g = torch.randint(7, len(vocab), (bs, sl), generator=gen).cuda()
    c = torch.randint(1, 30, (bs, sl), generator=gen).float().cuda()
    g[:, sl-sl//10:] = 5; c[:, sl-sl//10:] = 0
    with torch.no_grad():
        _ = sdpa(gene_token_indices=g, gene_counts=c)
    for h in hooks: h.remove()
    return k_acts, v_acts, mc


def attn_cos(q, k_orig, v_orig, k_rec, v_rec):
    B, H, S, D = k_orig.shape
    sc = 1.0 / math.sqrt(D)
    s_fp = torch.matmul(q, k_orig.transpose(-2,-1)) * sc
    s_q  = torch.matmul(q, k_rec.transpose(-2,-1)) * sc
    w_fp = F.softmax(s_fp, dim=-1)
    w_q  = F.softmax(s_q, dim=-1)
    o_fp = torch.matmul(w_fp, v_orig)
    o_q  = torch.matmul(w_q, v_rec)
    return F.cosine_similarity(o_fp.reshape(B,-1), o_q.reshape(B,-1), dim=-1).mean().item()


def vec_cos(a, b):
    B = a.shape[0]
    return F.cosine_similarity(a.reshape(B,-1).float(), b.reshape(B,-1).float(), dim=-1).mean().item()


def dequant_lm(indices, centroids, Pi, B, H, S, D, C, vector_norms=None):
    """Lloyd-Max dequantize + inverse rotation (no QJL)."""
    c_exp = centroids[None,:,None,None,:].expand(B,H,S,D,C)
    x_deq = torch.gather(c_exp, -1, indices.long().unsqueeze(-1)).squeeze(-1)
    # Norm correction before inverse rotation
    x_deq_norms = torch.norm(x_deq, dim=-1, keepdim=True)
    safe_norms = torch.where(x_deq_norms > 1e-10, x_deq_norms, torch.ones_like(x_deq_norms))
    x_deq = x_deq / safe_norms
    x_recon = torch.einsum("bhsd,hde->bhse", x_deq, Pi)
    if vector_norms is not None:
        x_recon = x_recon * vector_norms.unsqueeze(-1)
    return x_recon


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()

    print("=" * 70)
    print("TurboGene: Ablation Study & KIVI Baseline")
    print("=" * 70)

    k_acts, v_acts, mc = get_activations(args.checkpoint_dir)
    L, H, D = mc.num_layers, mc.num_heads, mc.embed_dim // mc.num_heads

    from turbogene.quantizer import TurboGeneQuantizer
    from turbogene.baselines import KIVIQuantizer

    # Calibrated quantizers
    q_svd = TurboGeneQuantizer(L, H, D, n_bits=3).cuda()
    q_svd.calibrate(k_acts, v_acts)

    q_rand = TurboGeneQuantizer(L, H, D, n_bits=3).cuda()
    q_rand.k_centroids.copy_(q_svd.k_centroids)
    q_rand.v_centroids.copy_(q_svd.v_centroids)
    # q_rand keeps its random rotations from init

    q_shared = TurboGeneQuantizer(L, H, D, n_bits=3).cuda()
    q_shared.rotations.copy_(q_svd.rotations)
    avg_c = (q_svd.k_centroids + q_svd.v_centroids) / 2
    q_shared.k_centroids.copy_(avg_c)
    q_shared.v_centroids.copy_(avg_c)

    kivi = KIVIQuantizer(n_bits=2)

    # Collect per-layer results
    abl = {}
    kivi_res = {}

    for layer_idx, k in k_acts:
        _, v = v_acts[layer_idx]
        B, Hh, S, Dd = k.shape
        C = q_svd.n_centroids
        qq = k[:, :, -1:, :]  # query = last position
        Pi = q_svd.rotations[layer_idx]

        # (a) Rotation only
        k_rot = torch.einsum("bhsd,hde->bhse", k.float(), Pi.transpose(-1,-2))
        k_unrot = torch.einsum("bhsd,hde->bhse", k_rot, Pi)
        v_rot = torch.einsum("bhsd,hde->bhse", v.float(), Pi.transpose(-1,-2))
        v_unrot = torch.einsum("bhsd,hde->bhse", v_rot, Pi)
        a_rot = attn_cos(qq.float(), k.float(), v.float(), k_unrot, v_unrot)
        k_rot_cos = vec_cos(k, k_unrot)

        # (b) Rotation + Lloyd-Max (no QJL)
        idx_k, _, _, vnk = q_svd.quantize_vector(k, layer_idx, True)
        idx_v, _, _, vnv = q_svd.quantize_vector(v, layer_idx, False)
        k_lm = dequant_lm(idx_k, q_svd.k_centroids[layer_idx], Pi, B, Hh, S, Dd, C, vnk)
        v_lm = dequant_lm(idx_v, q_svd.v_centroids[layer_idx], Pi, B, Hh, S, Dd, C, vnv)
        a_lm = attn_cos(qq.float(), k.float(), v.float(), k_lm, v_lm)
        k_lm_cos = vec_cos(k, k_lm)

        # (c) Full TurboGene
        ik, sk, mk, nk = q_svd.quantize_vector(k, layer_idx, True)
        iv, sv, mv, nv = q_svd.quantize_vector(v, layer_idx, False)
        k_full = q_svd.dequantize_vector(ik, sk, mk, nk, layer_idx, True)
        v_full = q_svd.dequantize_vector(iv, sv, mv, nv, layer_idx, False)
        a_full = attn_cos(qq.float(), k.float(), v.float(), k_full, v_full)
        k_full_cos = vec_cos(k, k_full)

        # (d) Random rotation
        ik_r, sk_r, mk_r, nk_r = q_rand.quantize_vector(k, layer_idx, True)
        iv_r, sv_r, mv_r, nv_r = q_rand.quantize_vector(v, layer_idx, False)
        k_rnd = q_rand.dequantize_vector(ik_r, sk_r, mk_r, nk_r, layer_idx, True)
        v_rnd = q_rand.dequantize_vector(iv_r, sv_r, mv_r, nv_r, layer_idx, False)
        a_rnd = attn_cos(qq.float(), k.float(), v.float(), k_rnd, v_rnd)
        k_rnd_cos = vec_cos(k, k_rnd)

        # (e) Shared LUT
        ik_s, sk_s, mk_s, nk_s = q_shared.quantize_vector(k, layer_idx, True)
        iv_s, sv_s, mv_s, nv_s = q_shared.quantize_vector(v, layer_idx, False)
        k_shr = q_shared.dequantize_vector(ik_s, sk_s, mk_s, nk_s, layer_idx, True)
        v_shr = q_shared.dequantize_vector(iv_s, sv_s, mv_s, nv_s, layer_idx, False)
        a_shr = attn_cos(qq.float(), k.float(), v.float(), k_shr, v_shr)
        k_shr_cos = vec_cos(k, k_shr)

        # KIVI
        ki_k, ks, kz = kivi.quantize_keys(k.float())
        ki_v, vs, vz = kivi.quantize_values(v.float())
        k_kv = kivi.dequantize_keys(ki_k, ks, kz)
        v_kv = kivi.dequantize_values(ki_v, vs, vz)
        a_kivi = attn_cos(qq.float(), k.float(), v.float(), k_kv, v_kv)
        k_kivi_cos = vec_cos(k, k_kv)

        abl[layer_idx] = {
            "rot":    {"attn": a_rot,  "k": k_rot_cos},
            "lm":     {"attn": a_lm,   "k": k_lm_cos},
            "full":   {"attn": a_full,  "k": k_full_cos},
            "rand":   {"attn": a_rnd,   "k": k_rnd_cos},
            "shared": {"attn": a_shr,   "k": k_shr_cos},
        }
        kivi_res[layer_idx] = {"attn": a_kivi, "k": k_kivi_cos}

    # ── Print results ──
    def mean_metric(d, key, sub):
        return np.mean([d[li][key][sub] for li in d])

    print("\n--- ABLATION: Pipeline Components ---")
    print(f"{'Config':25s} {'Attn cos':>10s} {'K cos':>10s}")
    for label, key in [("(a) Rotation only", "rot"), ("(b) + Lloyd-Max 3-bit", "lm"),
                        ("(c) + QJL 1-bit (full)", "full")]:
        a = mean_metric(abl, key, "attn")
        k = mean_metric(abl, key, "k")
        print(f"  {label:25s} {a:10.6f} {k:10.6f}")

    qjl_delta = mean_metric(abl,"full","attn") - mean_metric(abl,"lm","attn")
    print(f"  QJL improvement:        {qjl_delta:+10.6f}")

    print("\n--- ABLATION: Rotation Type ---")
    for label, key in [("Random rotation", "rand"), ("SVD rotation (ours)", "full")]:
        a = mean_metric(abl, key, "attn")
        print(f"  {label:25s} {a:10.6f}")
    svd_delta = mean_metric(abl,"full","attn") - mean_metric(abl,"rand","attn")
    print(f"  SVD improvement:        {svd_delta:+10.6f}")

    print("\n--- ABLATION: LUT Type ---")
    for label, key in [("Shared LUT", "shared"), ("Dual K/V LUT (ours)", "full")]:
        a = mean_metric(abl, key, "attn")
        print(f"  {label:25s} {a:10.6f}")
    lut_delta = mean_metric(abl,"full","attn") - mean_metric(abl,"shared","attn")
    print(f"  Dual LUT improvement:   {lut_delta:+10.6f}")

    # ── KIVI comparison ──
    print("\n--- BASELINE COMPARISON ---")
    kivi_mem = kivi.memory_bytes(4, 2048, L, H, D)
    tg_mem = q_svd.memory_bytes(4, 2048)
    kivi_a = np.mean([kivi_res[li]["attn"] for li in kivi_res])
    kivi_k = np.mean([kivi_res[li]["k"] for li in kivi_res])
    tg_a = mean_metric(abl, "full", "attn")
    tg_k = mean_metric(abl, "full", "k")

    print(f"\n{'Method':30s} {'Bits':>6s} {'Ratio':>7s} {'Attn cos':>10s} {'K cos':>10s}")
    print(f"{'-'*63}")
    print(f"{'FP16':30s} {'16.0':>6s} {'1.00x':>7s} {'1.000000':>10s} {'1.000000':>10s}")
    print(f"{'KIVI 2-bit asymmetric':30s} {kivi_mem['effective_bits']:6.1f} {kivi_mem['compression_ratio']:6.2f}x {kivi_a:10.6f} {kivi_k:10.6f}")
    print(f"{'TurboGene 3-bit (ours)':30s} {tg_mem['effective_bits_per_coord']:6.1f} {tg_mem['compression_ratio']:6.2f}x {tg_a:10.6f} {tg_k:10.6f}")

    delta_vs_kivi = tg_a - kivi_a
    print(f"\n  TurboGene vs KIVI delta: {delta_vs_kivi:+.6f} attention cos_sim")
    print(f"  (positive = TurboGene better)")

    # Per-layer
    print(f"\n  Per-layer: KIVI vs TurboGene (attention cos_sim)")
    print(f"  {'Layer':>6s} {'KIVI':>10s} {'TurboGene':>10s} {'Delta':>10s}")
    for li in sorted(kivi_res.keys()):
        kv = kivi_res[li]["attn"]
        tg = abl[li]["full"]["attn"]
        print(f"  {li:6d} {kv:10.6f} {tg:10.6f} {tg-kv:+10.6f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
