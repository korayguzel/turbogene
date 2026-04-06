"""
TurboGene: Generative Decode Quality Benchmark

Tests whether KV cache quantization degrades downstream biological quality
when K/V are used in their quantized-dequantized form (simulating decode).

Three validation axes:
  1. Embedding quality: cosine similarity, classification accuracy
  2. Gene prediction quality: logit agreement, count prediction correlation
  3. Generated cell quality: marker gene distributions, gene-gene correlations

All gene_logit metrics are computed online (batch-by-batch) to avoid storing
the full (N, 2048, 23830) logit tensor in RAM.

Usage:
    python scripts/generative_decode_benchmark.py --checkpoint-dir checkpoints/tf_sapiens
"""

import gc, sys, time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def load_models(ckpt):
    """Load original SDPA model and TurboGene quantized model."""
    from scripts.test_sdpa_e2e import load_original_model
    from turbogene.sdpa_model import SDPATranscriptformer
    from turbogene.quantizer import TurboGeneQuantizer
    from turbogene.quantized_model import QuantizedTranscriptformer

    orig, vocab = load_original_model(ckpt)
    orig = orig.to("cuda", torch.float32).eval()
    sdpa = SDPATranscriptformer(orig).to("cuda", torch.float32).eval()
    mc = orig.model_config
    nh, hd = mc.num_heads, mc.embed_dim // mc.num_heads
    sl = mc.seq_len + mc.aux_len

    q = TurboGeneQuantizer(mc.num_layers, nh, hd, n_bits=3).cuda()
    k_a, v_a = [], []
    def mh(li):
        def hook(mod, args):
            inp = args[0]; B = inp.size(0)
            k_a.append((li, mod.linears[1](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2).detach()))
            v_a.append((li, mod.linears[2](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2).detach()))
        return hook
    hooks = [l.self_attn.register_forward_pre_hook(mh(i))
             for i, l in enumerate(sdpa.transformer_encoder.encoder_layers)]
    gen = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        _ = sdpa(gene_token_indices=torch.randint(7, len(vocab), (2, sl), generator=gen).cuda(),
                 gene_counts=torch.randint(1, 30, (2, sl), generator=gen).float().cuda())
    for h in hooks: h.remove()
    q.calibrate(k_a, v_a)
    del k_a, v_a, orig

    qm = QuantizedTranscriptformer(sdpa, q).to("cuda", torch.float32).eval()
    return sdpa, qm, vocab, mc


def collect_embeddings(model, g, c, bs, forward_fn):
    """Run model batch-by-batch with embed=True, collect only embeddings."""
    N = g.size(0)
    all_embs = []
    for i in range(0, N, bs):
        gi, ci = g[i:i+bs].cuda(), c[i:i+bs].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            r = forward_fn(gi, ci, True)  # embed=True
        all_embs.append(r["embeddings"].detach().cpu())
        del r
    return torch.cat(all_embs, dim=0)


def collect_mu(model, g, c, bs, forward_fn):
    """Run model batch-by-batch with embed=False, collect mu and mask."""
    N = g.size(0)
    all_mu, all_mask = [], []
    for i in range(0, N, bs):
        gi, ci = g[i:i+bs].cuda(), c[i:i+bs].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            r = forward_fn(gi, ci, False)  # embed=False
        all_mu.append(r["mu"].detach().cpu())
        all_mask.append(r["mask"].detach().cpu())
        del r
    return torch.cat(all_mu, dim=0), torch.cat(all_mask, dim=0)


def compute_logit_metrics_online(sdpa, qm, g, c, bs=2):
    """Compute gene logit KL divergence and top-1 agreement batch-by-batch.

    Never stores full (B, S, 23830) logit tensors — computes metrics per batch
    and accumulates running statistics.
    """
    N = g.size(0)
    kl_sum, kl_count = 0.0, 0
    agree_sum, agree_count = 0, 0
    kl_max = 0.0

    for i in range(0, N, bs):
        gi, ci = g[i:i+bs].cuda(), c[i:i+bs].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            r_orig = sdpa(gene_token_indices=gi, gene_counts=ci, embed=False)
            r_tg = qm.forward_with_decode_quantization(gi, ci, embed=False)

        if "gene_logit" not in r_orig or "gene_logit" not in r_tg:
            del r_orig, r_tg
            return None

        lo = r_orig["gene_logit"].float().cpu()
        lt = r_tg["gene_logit"].float().cpu()
        mask = r_tg["mask"].bool().cpu()
        del r_orig, r_tg

        # Top-1 agreement
        top1_o = lo.argmax(dim=-1)
        top1_t = lt.argmax(dim=-1)
        valid = mask
        agree_sum += int((top1_o == top1_t)[valid].sum())
        agree_count += int(valid.sum())

        # KL divergence (sample positions for speed)
        b_size, seq_len = mask.shape
        for bi in range(b_size):
            valid_pos = mask[bi].nonzero(as_tuple=True)[0]
            # Sample up to 200 positions per cell
            if len(valid_pos) > 200:
                idx = valid_pos[torch.randperm(len(valid_pos))[:200]]
            else:
                idx = valid_pos
            for j in idx:
                p = F.softmax(lo[bi, j], dim=-1)
                q = F.softmax(lt[bi, j], dim=-1)
                kl = F.kl_div(q.log(), p, reduction="sum").item()
                kl_sum += kl
                kl_count += 1
                if kl > kl_max:
                    kl_max = kl

        del lo, lt, mask
        print("     Processed %d/%d cells" % (min(i+bs, N), N))

    if kl_count == 0:
        return None

    return {
        "kl_mean": kl_sum / kl_count,
        "kl_max": kl_max,
        "top1_agreement": agree_sum / max(agree_count, 1),
    }


def sample_counts_online(model, g, c, bs, forward_fn, seed=42):
    """Sample ZTP counts from mu predictions, batch by batch."""
    torch.manual_seed(seed)
    N = g.size(0)
    all_counts, all_masks = [], []
    for i in range(0, N, bs):
        gi, ci = g[i:i+bs].cuda(), c[i:i+bs].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            r = forward_fn(gi, ci, False)  # embed=False

        mu = r["mu"].float().cpu().clamp(min=1e-5)
        mask = r["mask"].cpu()
        del r

        # ZTP sampling (clamp rate to avoid negative values)
        p_zero = torch.exp(-mu)
        u = torch.rand_like(mu) * (1 - p_zero) + p_zero
        t = -torch.log(u)
        rate = (mu - t).clamp(min=0.0)
        counts = (1 + torch.poisson(rate)).clamp(0, 30) * mask.float()

        all_counts.append(counts)
        all_masks.append(mask)
        del mu

    return torch.cat(all_counts, dim=0), torch.cat(all_masks, dim=0)


def knn_classify(embs, labels, k=15):
    le = LabelEncoder(); y = le.fit_transform(labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc, f1, ari = [], [], []
    for tr, te in skf.split(embs, y):
        kn = KNeighborsClassifier(n_neighbors=min(k, len(tr)-1))
        kn.fit(embs[tr], y[tr]); p = kn.predict(embs[te])
        acc.append(accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="weighted"))
        ari.append(adjusted_rand_score(y[te], p))
    return np.mean(acc), np.std(acc), np.mean(f1), np.mean(ari)


def compare_marker_genes(counts_orig, counts_tg, labels, markers):
    results = {}
    for mtype, patterns in markers.items():
        mask = np.array([any(p.lower() in str(l).lower() for p in patterns) for l in labels])
        if mask.sum() < 5:
            continue
        orig_mean = counts_orig[mask].float().mean(dim=0).numpy()
        tg_mean = counts_tg[mask].float().mean(dim=0).numpy()
        r, _ = stats.pearsonr(orig_mean, tg_mean)
        cos = float(np.dot(orig_mean, tg_mean) / (np.linalg.norm(orig_mean) * np.linalg.norm(tg_mean) + 1e-8))
        results[mtype] = {"n": int(mask.sum()), "pearson": r, "cosine": cos}
    return results


def compare_gene_correlations(counts_orig, counts_tg, top_k=50):
    var_orig = counts_orig.float().var(dim=0).numpy()
    top_idx = np.argsort(var_orig)[-top_k:]
    co = counts_orig[:, top_idx].float().numpy()
    ct = counts_tg[:, top_idx].float().numpy()
    corr_orig = np.corrcoef(co.T)
    corr_tg = np.corrcoef(ct.T)
    triu = np.triu_indices(top_k, k=1)
    flat_orig = corr_orig[triu]
    flat_tg = corr_tg[triu]
    valid = ~(np.isnan(flat_orig) | np.isnan(flat_tg))
    if valid.sum() < 10:
        return 0.0, 0.0
    r, _ = stats.pearsonr(flat_orig[valid], flat_tg[valid])
    rmse = float(np.sqrt(np.mean((flat_orig[valid] - flat_tg[valid]) ** 2)))
    return r, rmse


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--n-cells", type=int, default=500,
                        help="Number of cells for generation comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("TurboGene: Generative Decode Quality Benchmark")
    print("=" * 70)
    print("GPU: %s" % torch.cuda.get_device_name(0))

    print("\n1. Loading models...")
    sdpa, qm, vocab, mc = load_models(args.checkpoint_dir)
    sl = mc.seq_len + mc.aux_len

    print("\n2. Loading datasets...")
    datasets = {}
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
        datasets[name] = {"g": g, "c": c, "labels": labels, "n": len(vi)}
        print("   %s: %d cells, %d types" % (name, len(vi), len(set(labels))))

    dname = list(datasets.keys())[0]
    d = datasets[dname]
    n = min(args.n_cells, d["n"])
    g_sub, c_sub = d["g"][:n], d["c"][:n]
    labels_sub = d["labels"][:n]
    print("\n   Using %s: %d cells for decode quality test" % (dname, n))

    def orig_fwd(gi, ci, emb):
        return sdpa(gene_token_indices=gi, gene_counts=ci, embed=emb)
    def decode_fwd(gi, ci, emb):
        return qm.forward_with_decode_quantization(gi, ci, embed=emb)
    def prefill_fwd(gi, ci, emb):
        return qm.forward_memory_efficient(gi, ci, embed=emb)

    # ══════════════════════════════════════════════════════════════
    # A. EMBEDDING QUALITY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("A. EMBEDDING QUALITY (decode-quantized K/V)")
    print("=" * 70)

    reset()
    print("   Running original model (embeddings)...")
    orig_embs = collect_embeddings(sdpa, g_sub, c_sub, 4, orig_fwd)

    reset()
    print("   Running TurboGene decode (embeddings)...")
    tg_embs = collect_embeddings(qm, g_sub, c_sub, 4, decode_fwd)

    reset()
    print("   Running TurboGene prefill (embeddings)...")
    pf_embs = collect_embeddings(qm, g_sub, c_sub, 4, prefill_fwd)

    cos_decode = F.cosine_similarity(orig_embs, tg_embs, dim=-1)
    cos_prefill = F.cosine_similarity(orig_embs, pf_embs, dim=-1)
    print("\n   Embedding cosine similarity:")
    print("     Prefill (no quant in attn): mean=%.6f min=%.6f" % (cos_prefill.mean(), cos_prefill.min()))
    print("     Decode (quant K/V in attn): mean=%.6f min=%.6f" % (cos_decode.mean(), cos_decode.min()))

    acc_o = acc_d = acc_p = f1_o = f1_d = ari_o = ari_d = 0
    if len(set(labels_sub)) >= 3:
        acc_o, std_o, f1_o, ari_o = knn_classify(orig_embs.numpy(), labels_sub)
        acc_d, std_d, f1_d, ari_d = knn_classify(tg_embs.numpy(), labels_sub)
        acc_p, std_p, f1_p, ari_p = knn_classify(pf_embs.numpy(), labels_sub)
        print("\n   Cell type classification (kNN k=15, 5-fold CV):")
        print("     %-35s %8s %8s %8s" % ("Method", "Acc", "F1", "ARI"))
        print("     %-35s %8.4f %8.4f %8.4f" % ("Original (FP16)", acc_o, f1_o, ari_o))
        print("     %-35s %8.4f %8.4f %8.4f" % ("TurboGene prefill (no quant attn)", acc_p, f1_p, ari_p))
        print("     %-35s %8.4f %8.4f %8.4f" % ("TurboGene decode (quant K/V attn)", acc_d, f1_d, ari_d))
        print("     Delta (decode vs original):       %+8.4f %+8.4f %+8.4f" % (acc_d-acc_o, f1_d-f1_o, ari_d-ari_o))

    del pf_embs

    # ══════════════════════════════════════════════════════════════
    # B. COUNT PREDICTION QUALITY (run with embed=False for mu)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("B. COUNT PREDICTION QUALITY (decode-quantized K/V)")
    print("=" * 70)

    reset()
    print("   Running original model (mu predictions)...")
    orig_mu, orig_mask = collect_mu(sdpa, g_sub, c_sub, 4, orig_fwd)
    print("   Running TurboGene decode (mu predictions)...")
    tg_mu, tg_mask = collect_mu(qm, g_sub, c_sub, 4, decode_fwd)

    valid = tg_mask.bool().flatten()
    o_flat = orig_mu.float().flatten()[valid].numpy()
    t_flat = tg_mu.float().flatten()[valid].numpy()

    # Filter out constant/zero values for correlation
    nz = (np.abs(o_flat) > 1e-8) | (np.abs(t_flat) > 1e-8)
    if nz.sum() > 10:
        pearson, _ = stats.pearsonr(o_flat[nz], t_flat[nz])
        spearman, _ = stats.spearmanr(o_flat[nz], t_flat[nz])
    else:
        pearson, spearman = float('nan'), float('nan')
    mse = float(np.mean((o_flat - t_flat) ** 2))
    rel_err = float(np.mean(np.abs(o_flat - t_flat) / (np.abs(o_flat) + 1e-6)))

    print("\n   Count prediction (mu) comparison:")
    print("     Pearson correlation:  %.6f" % pearson)
    print("     Spearman correlation: %.6f" % spearman)
    print("     MSE:                  %.6f" % mse)
    print("     Mean relative error:  %.6f" % rel_err)

    # Gene logit comparison (online — never stores full logits)
    reset()
    print("\n   Running gene logit comparison (online)...")
    n_logit = min(50, n)  # Use 50 cells for logit comparison
    logit_metrics = compute_logit_metrics_online(sdpa, qm, g_sub[:n_logit], c_sub[:n_logit], bs=2)

    if logit_metrics is not None:
        print("\n   Gene logit KL divergence (orig || decode):")
        print("     Mean:   %.6f" % logit_metrics["kl_mean"])
        print("     Max:    %.6f" % logit_metrics["kl_max"])
        print("     Top-1 gene agreement: %.4f" % logit_metrics["top1_agreement"])

    # ══════════════════════════════════════════════════════════════
    # C. GENERATED CELL QUALITY (sampled from mu, no gene_logit needed)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("C. GENERATED CELL QUALITY (ZTP-sampled counts)")
    print("=" * 70)

    reset()
    print("   Sampling cells from original model predictions...")
    orig_counts, orig_cmask = sample_counts_online(sdpa, g_sub, c_sub, 4, orig_fwd, seed=42)
    print("   Sampling cells from TurboGene decode predictions...")
    tg_counts, tg_cmask = sample_counts_online(qm, g_sub, c_sub, 4, decode_fwd, seed=42)

    # C1: Marker gene expression
    markers = {
        "T cell": ["T cell", "alpha-beta T"],
        "B cell": ["B cell", "naive B"],
        "monocyte": ["monocyte", "classical monocyte"],
        "NK": ["natural killer", "NK"],
        "erythrocyte": ["erythrocyte"],
        "hepatocyte": ["hepatocyte", "hepatic stellate"],
    }
    print("\n   C1. Marker gene expression distributions:")
    marker_results = compare_marker_genes(orig_counts, tg_counts, labels_sub, markers)
    if marker_results:
        print("   %-25s %5s %10s %10s" % ("Type", "N", "Pearson", "Cosine"))
        for mtype, mr in marker_results.items():
            print("   %-25s %5d %10.6f %10.6f" % (mtype, mr["n"], mr["pearson"], mr["cosine"]))
    else:
        print("   (No marker types found with sufficient cells)")

    # C2: Gene-gene correlation
    print("\n   C2. Gene-gene correlation structure (top-50 variable positions):")
    corr_r, corr_rmse = compare_gene_correlations(orig_counts, tg_counts, top_k=50)
    print("     Correlation matrix Pearson: %.6f" % corr_r)
    print("     Correlation matrix RMSE:    %.6f" % corr_rmse)

    # C3: Overall count distribution
    valid_m = orig_cmask.bool()
    orig_flat = orig_counts[valid_m].float()
    tg_flat = tg_counts[valid_m].float()
    count_ks, count_p = stats.ks_2samp(orig_flat.numpy(), tg_flat.numpy())
    print("\n   C3. Overall count distribution (KS test):")
    print("     KS statistic: %.6f" % count_ks)
    print("     p-value:      %.6f" % count_p)
    print("     Mean count orig:  %.3f +/- %.3f" % (orig_flat.mean(), orig_flat.std()))
    print("     Mean count TG:    %.3f +/- %.3f" % (tg_flat.mean(), tg_flat.std()))

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DECODE QUALITY BENCHMARK SUMMARY")
    print("=" * 70)
    print("   Dataset:          %s (%d cells)" % (dname, n))
    print("   Embedding cos:    %.6f (decode) vs %.6f (prefill)" % (cos_decode.mean(), cos_prefill.mean()))
    if len(set(labels_sub)) >= 3:
        print("   Classification:   %.4f (decode) vs %.4f (original), delta=%+.4f" % (acc_d, acc_o, acc_d-acc_o))
    print("   Count Pearson:    %.6f" % pearson)
    print("   Count Spearman:   %.6f" % spearman)
    if logit_metrics is not None:
        print("   Gene logit KL:    %.6f (mean)" % logit_metrics["kl_mean"])
        print("   Top-1 agreement:  %.4f" % logit_metrics["top1_agreement"])
    print("   Gene-gene corr:   %.6f (Pearson of correlation matrices)" % corr_r)

    print("\n   VERDICT: ", end="")
    if cos_decode.mean() > 0.99 and pearson > 0.99:
        print("Decode quantization preserves biological quality (practically lossless)")
    elif cos_decode.mean() > 0.95 and pearson > 0.95:
        print("Decode quantization causes minor but measurable quality reduction")
    else:
        print("Decode quantization causes significant quality reduction")

    print("=" * 70)


if __name__ == "__main__":
    main()
