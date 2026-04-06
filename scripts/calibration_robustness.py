"""
TurboGene: Calibration Robustness Ablation

Tests how the number of calibration samples affects quantization quality.
Reports decode cosine similarity and classification accuracy for n=2, 10, 50, 100.

Usage:
    python scripts/calibration_robustness.py --checkpoint-dir checkpoints/tf_sapiens
"""

import gc, math, sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def load_base(ckpt):
    """Load original model and SDPA wrapper (no quantizer yet)."""
    from scripts.test_sdpa_e2e import load_original_model
    from turbogene.sdpa_model import SDPATranscriptformer

    orig, vocab = load_original_model(ckpt)
    orig = orig.to("cuda", torch.float32).eval()
    sdpa = SDPATranscriptformer(orig).to("cuda", torch.float32).eval()
    mc = orig.model_config
    del orig
    return sdpa, vocab, mc


def collect_activations(sdpa, vocab, mc, n_samples, seed=42):
    """Collect K/V activations from n_samples random inputs for calibration."""
    sl = mc.seq_len + mc.aux_len
    k_acts, v_acts = [], []

    # Subsample positions in hooks to keep CPU memory bounded
    pos_subsample = 256

    def mh(li):
        def hook(mod, args):
            inp = args[0]; B = inp.size(0)
            k = mod.linears[1](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2).detach()
            v = mod.linears[2](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2).detach()
            # Subsample positions to save memory
            S = k.shape[2]
            if S > pos_subsample:
                idx = torch.randperm(S, device=k.device)[:pos_subsample].sort().values
                k = k[:, :, idx, :]
                v = v[:, :, idx, :]
            k_acts.append((li, k.cpu()))
            v_acts.append((li, v.cpu()))
        return hook

    hooks = [l.self_attn.register_forward_pre_hook(mh(i))
             for i, l in enumerate(sdpa.transformer_encoder.encoder_layers)]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    bs = min(n_samples, 4)
    for start in range(0, n_samples, bs):
        end = min(start + bs, n_samples)
        actual_bs = end - start
        with torch.no_grad():
            _ = sdpa(
                gene_token_indices=torch.randint(7, len(vocab), (actual_bs, sl), generator=gen).cuda(),
                gene_counts=torch.randint(1, 30, (actual_bs, sl), generator=gen).float().cuda(),
            )

    for h in hooks:
        h.remove()

    # Merge activations per layer (they were appended per-batch)
    merged_k, merged_v = {}, {}
    for li, tensor in k_acts:
        if li not in merged_k:
            merged_k[li] = []
        merged_k[li].append(tensor)
    for li, tensor in v_acts:
        if li not in merged_v:
            merged_v[li] = []
        merged_v[li].append(tensor)

    k_merged = [(li, torch.cat(tensors, dim=0)) for li, tensors in sorted(merged_k.items())]
    v_merged = [(li, torch.cat(tensors, dim=0)) for li, tensors in sorted(merged_v.items())]

    return k_merged, v_merged


def build_quantized_model(sdpa, mc, k_acts, v_acts):
    """Build QuantizedTranscriptformer with given calibration data."""
    from turbogene.quantizer import TurboGeneQuantizer
    from turbogene.quantized_model import QuantizedTranscriptformer

    nh, hd = mc.num_heads, mc.embed_dim // mc.num_heads
    # Calibrate on CPU first, then move to CUDA
    q = TurboGeneQuantizer(mc.num_layers, nh, hd, n_bits=3)
    q.calibrate(k_acts, v_acts)
    q = q.cuda()
    qm = QuantizedTranscriptformer(sdpa, q).to("cuda", torch.float32).eval()
    return qm


def measure_decode_cosine(qm, k_acts, v_acts, mc):
    """Measure attention cosine similarity with quantized K/V (decode simulation)."""
    cos_sims = []
    for (li_k, k_cpu), (li_v, v_cpu) in zip(k_acts, v_acts):
        # Process in small batches to avoid OOM
        k = k_cpu[:2].cuda()  # Use only first 2 samples for cosine measurement
        v = v_cpu[:2].cuda()
        B, H, S, D = k.shape
        qq = k[:, :, -1:, :]

        ik, sk, mk, nk = qm.quantizer.quantize_vector(k, li_k, is_key=True)
        iv, sv, mv, nv = qm.quantizer.quantize_vector(v, li_v, is_key=False)
        k_rec = qm.quantizer.dequantize_vector(ik, sk, mk, nk, li_k, is_key=True)
        v_rec = qm.quantizer.dequantize_vector(iv, sv, mv, nv, li_v, is_key=False)

        scale = 1.0 / math.sqrt(D)
        s_fp = torch.matmul(qq.float(), k.float().transpose(-2, -1)) * scale
        s_q = torch.matmul(qq.float(), k_rec.float().transpose(-2, -1)) * scale
        w_fp = F.softmax(s_fp, dim=-1)
        w_q = F.softmax(s_q, dim=-1)
        o_fp = torch.matmul(w_fp, v.float())
        o_q = torch.matmul(w_q, v_rec.float())
        cos = F.cosine_similarity(o_fp.reshape(B, -1), o_q.reshape(B, -1), dim=-1).mean().item()
        cos_sims.append(cos)
        del k, v, qq, ik, sk, mk, iv, sv, mv, k_rec, v_rec

    return float(np.mean(cos_sims)), float(np.min(cos_sims))


def measure_classification(qm, g, c, labels, bs=4):
    """Measure classification accuracy with decode-quantized forward."""
    N = g.size(0)
    all_embs = []
    for i in range(0, N, bs):
        gi, ci = g[i:i+bs].cuda(), c[i:i+bs].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            r = qm.forward_with_decode_quantization(gi, ci, embed=True)
        all_embs.append(r["embeddings"].cpu())
        del r

    embs = torch.cat(all_embs, dim=0).numpy()
    le = LabelEncoder(); y = le.fit_transform(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for tr, te in skf.split(embs, y):
        kn = KNeighborsClassifier(n_neighbors=min(15, len(tr)-1))
        kn.fit(embs[tr], y[tr]); p = kn.predict(embs[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="weighted"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()

    print("=" * 70)
    print("TurboGene: Calibration Robustness Ablation")
    print("=" * 70)
    print("GPU: %s" % torch.cuda.get_device_name(0))

    print("\n1. Loading base model...")
    sdpa, vocab, mc = load_base(args.checkpoint_dir)
    sl = mc.seq_len + mc.aux_len

    print("\n2. Loading evaluation dataset...")
    eval_data = None
    for name, path in [("cellxgene_10k", "data/benchmark/cellxgene_10k_vocabonly.h5ad"),
                       ("cardiac_1k", "transcriptformer/test/data/human_val.h5ad")]:
        if not Path(path).exists():
            continue
        adata = ad.read_h5ad(path)
        from turbogene.data_utils import fast_tokenize
        g, c, vi = fast_tokenize(adata, vocab, sl)
        if g is None:
            continue
        ct_col = "cell_type" if "cell_type" in adata.obs.columns else "cell_state"
        labels = adata.obs[ct_col].values[vi]
        n_val = min(500, len(vi))
        eval_data = {"name": name, "g": g[:n_val], "c": c[:n_val],
                     "labels": labels[:n_val], "n": n_val}
        print("   %s: %d cells, %d types" % (name, n_val, len(set(labels[:n_val]))))
        break

    if eval_data is None:
        print("ERROR: No evaluation dataset found")
        return

    n_samples_list = [2, 5, 10, 25]
    print("\n3. Testing calibration with n = %s samples..." % str(n_samples_list))

    results = []
    for n_cal in n_samples_list:
        print("\n   --- n_calibration = %d ---" % n_cal)
        reset()

        print("   Collecting activations...")
        k_acts, v_acts = collect_activations(sdpa, vocab, mc, n_cal, seed=42)

        print("   Calibrating quantizer...")
        qm = build_quantized_model(sdpa, mc, k_acts, v_acts)

        print("   Measuring decode cosine similarity...")
        mean_cos, min_cos = measure_decode_cosine(qm, k_acts, v_acts, mc)

        print("   Measuring classification accuracy...")
        acc, acc_std, f1 = measure_classification(
            qm, eval_data["g"], eval_data["c"], eval_data["labels"]
        )

        results.append({
            "n_cal": n_cal,
            "mean_cos": mean_cos,
            "min_cos": min_cos,
            "acc": acc,
            "acc_std": acc_std,
            "f1": f1,
        })

        print("   Decode cos: mean=%.6f min=%.6f" % (mean_cos, min_cos))
        print("   Classification: acc=%.4f+/-%.3f f1=%.4f" % (acc, acc_std, f1))

        del qm, k_acts, v_acts

    # Original model baseline
    print("\n   --- Original (no quantization) ---")
    reset()
    N = eval_data["n"]
    orig_embs = []
    for i in range(0, N, 4):
        gi, ci = eval_data["g"][i:i+4].cuda(), eval_data["c"][i:i+4].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            r = sdpa(gene_token_indices=gi, gene_counts=ci, embed=True)
        orig_embs.append(r["embeddings"].cpu())
    orig_embs = torch.cat(orig_embs, dim=0).numpy()
    le = LabelEncoder(); y = le.fit_transform(eval_data["labels"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    orig_accs = []
    for tr, te in skf.split(orig_embs, y):
        kn = KNeighborsClassifier(n_neighbors=min(15, len(tr)-1))
        kn.fit(orig_embs[tr], y[tr]); p = kn.predict(orig_embs[te])
        orig_accs.append(accuracy_score(y[te], p))
    orig_acc = float(np.mean(orig_accs))
    orig_std = float(np.std(orig_accs))
    print("   Original: acc=%.4f+/-%.3f" % (orig_acc, orig_std))

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION ROBUSTNESS SUMMARY")
    print("=" * 70)
    print("\n   %-10s %12s %12s %10s %10s %10s" % (
        "n_cal", "Attn cos", "Min cos", "Acc", "Acc std", "F1"))
    print("   " + "-" * 64)
    print("   %-10s %12s %12s %10.4f %10.3f %10s" % (
        "Original", "1.000000", "1.000000", orig_acc, orig_std, "---"))
    for r in results:
        print("   %-10d %12.6f %12.6f %10.4f %10.3f %10.4f" % (
            r["n_cal"], r["mean_cos"], r["min_cos"], r["acc"], r["acc_std"], r["f1"]))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
