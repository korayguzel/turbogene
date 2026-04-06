"""
TurboGene Phase 3: Downstream Task Validation and Throughput Benchmark

1. Cell type classification (kNN, 5-fold CV) on human + mouse datasets
2. Embedding cosine similarity (orig vs TurboGene)
3. Throughput: cells/second comparison
"""

import gc
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_all_models(checkpoint_dir):
    from scripts.test_sdpa_e2e import load_original_model
    from turbogene.sdpa_model import SDPATranscriptformer
    from turbogene.quantizer import TurboGeneQuantizer
    from turbogene.quantized_model import QuantizedTranscriptformer

    orig, vocab = load_original_model(checkpoint_dir)
    orig = orig.to("cuda", torch.float32).eval()
    sdpa = SDPATranscriptformer(orig).to("cuda", torch.float32).eval()

    mc = orig.model_config
    nh, hd = mc.num_heads, mc.embed_dim // mc.num_heads
    sl = mc.seq_len + mc.aux_len
    quantizer = TurboGeneQuantizer(mc.num_layers, nh, hd, n_bits=3).cuda()

    k_acts, v_acts = [], []
    def mh(li):
        def hook(mod, args):
            inp = args[0]; B = inp.size(0)
            k_acts.append((li, mod.linears[1](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
            v_acts.append((li, mod.linears[2](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
        return hook
    hooks = [layer.self_attn.register_forward_pre_hook(mh(i)) for i, layer in enumerate(sdpa.transformer_encoder.encoder_layers)]
    gen = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        _ = sdpa(gene_token_indices=torch.randint(7, len(vocab), (2, sl), generator=gen).cuda(),
                 gene_counts=torch.randint(1, 30, (2, sl), generator=gen).float().cuda())
    for h in hooks: h.remove()
    quantizer.calibrate(k_acts, v_acts)
    del k_acts, v_acts

    qmodel = QuantizedTranscriptformer(sdpa, quantizer).to("cuda", torch.float32).eval()
    return orig, qmodel, vocab, mc


def prepare_data(path, vocab, max_len):
    adata = ad.read_h5ad(path)
    gene_col = "ensembl_id" if "ensembl_id" in adata.var.columns else adata.var.index.name
    gene_ids = adata.var[gene_col].values if gene_col in adata.var.columns else adata.var.index.values
    vmap = {g: vocab[g] for g in gene_ids if g in vocab}

    label_col = next((c for c in ["cell_type", "cell_state"] if c in adata.obs.columns), None)
    if not label_col:
        return None
    labels = adata.obs[label_col].values

    X = adata.raw.X if adata.raw is not None else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    pad_id = vocab["[PAD]"]
    all_g, all_c, valid_labels = [], [], []

    for ci in range(X.shape[0]):
        row = X[ci].flatten()
        nz = row > 0
        genes_nz = gene_ids[nz]
        counts_nz = row[nz]
        inv = [i for i, g in enumerate(genes_nz) if g in vmap]
        if len(inv) < 10:
            continue

        gi = np.array([vmap[genes_nz[i]] for i in inv])
        cv = np.clip(counts_nz[inv].astype(np.float32), 0, 30)

        if len(gi) > max_len:
            sel = np.random.choice(len(gi), max_len, replace=False)
            gi, cv = gi[sel], cv[sel]

        pad_n = max_len - len(gi)
        if pad_n > 0:
            gi = np.concatenate([gi, np.full(pad_n, pad_id, dtype=np.int64)])
            cv = np.concatenate([cv, np.zeros(pad_n, dtype=np.float32)])

        all_g.append(gi)
        all_c.append(cv)
        valid_labels.append(labels[ci])

    if len(all_g) == 0:
        return None
    return (torch.tensor(np.stack(all_g), dtype=torch.long),
            torch.tensor(np.stack(all_c), dtype=torch.float32),
            np.array(valid_labels), label_col)


def get_embeddings(model, g, c, bs, efficient=False, is_original=False):
    from transcriptformer.data.dataclasses import BatchData
    N = g.size(0)
    embs = []
    for i in range(0, N, bs):
        gi, ci = g[i:i+bs].cuda(), c[i:i+bs].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            if efficient:
                out = model.forward_memory_efficient(gi, ci, embed=True)
            elif is_original:
                out = model(BatchData(gene_token_indices=gi, gene_counts=ci, aux_token_indices=None), embed=True)
            else:
                out = model(gene_token_indices=gi, gene_counts=ci, embed=True)
        embs.append(out["embeddings"].cpu())
    return torch.cat(embs, dim=0).numpy()


def knn_cv(embs, labels, k=15, n_splits=5):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, aris = [], [], []
    for tr, te in skf.split(embs, y):
        knn = KNeighborsClassifier(n_neighbors=min(k, len(tr)-1))
        knn.fit(embs[tr], y[tr])
        pred = knn.predict(embs[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="weighted"))
        aris.append(adjusted_rand_score(y[te], pred))
    return {"acc": np.mean(accs), "acc_std": np.std(accs),
            "f1": np.mean(f1s), "f1_std": np.std(f1s),
            "ari": np.mean(aris), "ari_std": np.std(aris)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {gpu_name} ({gpu_mb:.0f} MB)")

    print("\n1. Loading models and calibrating...")
    orig, qmodel, vocab, mc = load_all_models(args.checkpoint_dir)
    sl = mc.seq_len + mc.aux_len

    print("\n2. Preparing datasets...")
    datasets = {}
    for name, path in [("human", "transcriptformer/test/data/human_val.h5ad"),
                        ("mouse", "transcriptformer/test/data/mouse_val.h5ad")]:
        result = prepare_data(path, vocab, sl)
        if result:
            datasets[name] = result
            print(f"   {name}: {result[0].size(0)} cells, {len(set(result[2]))} types (col={result[3]})")

    # ── Classification ──
    print("\n" + "=" * 70)
    print("3. Cell Type Classification (kNN k=15, 5-fold CV)")
    print("=" * 70)

    cls_results = {}
    for ds_name, (g, c, labels, _) in datasets.items():
        print(f"\n   {ds_name}:")
        for mname, model, bs, eff, is_orig in [
            ("Original (batch=4)", orig, 4, False, True),
            ("TurboGene (batch=16)", qmodel, 16, True, False),
        ]:
            reset()
            m = model.half()
            embs = get_embeddings(m, g, c, bs=bs, efficient=eff, is_original=is_orig)
            model.float()
            metrics = knn_cv(embs, labels)
            cls_results[(ds_name, mname)] = metrics
            print(f"     {mname:28s}: Acc={metrics['acc']:.4f}+-{metrics['acc_std']:.3f} "
                  f"F1={metrics['f1']:.4f}+-{metrics['f1_std']:.3f} "
                  f"ARI={metrics['ari']:.4f}+-{metrics['ari_std']:.3f}")

    # ── Embedding Similarity ──
    print("\n" + "=" * 70)
    print("4. Embedding Similarity")
    print("=" * 70)

    emb_sims = {}
    for ds_name, (g, c, labels, _) in datasets.items():
        reset()
        e_orig = get_embeddings(orig.half(), g, c, bs=4, efficient=False, is_original=True)
        orig.float()
        e_turbo = get_embeddings(qmodel.half(), g, c, bs=16, efficient=True)
        qmodel.float()
        cos = F.cosine_similarity(torch.tensor(e_orig), torch.tensor(e_turbo), dim=-1)
        emb_sims[ds_name] = {"mean": cos.mean().item(), "min": cos.min().item(), "std": cos.std().item()}
        print(f"   {ds_name}: mean_cos={cos.mean().item():.6f} min={cos.min().item():.6f}")

    # ── Throughput ──
    print("\n" + "=" * 70)
    print("5. Throughput Benchmark")
    print("=" * 70)

    g, c = datasets["human"][0], datasets["human"][1]

    # Original
    print("\n   Original model:")
    del qmodel; reset()
    orig_hp = orig.half()
    from transcriptformer.data.dataclasses import BatchData

    orig_tp = {}
    for bs in [1, 2, 4]:
        try:
            reset()
            for _ in range(2):
                b = BatchData(gene_token_indices=g[:bs].cuda(), gene_counts=c[:bs].cuda(), aux_token_indices=None)
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    _ = orig_hp(b, embed=True)
            torch.cuda.synchronize()
            times = []
            for _ in range(5):
                b = BatchData(gene_token_indices=g[:bs].cuda(), gene_counts=c[:bs].cuda(), aux_token_indices=None)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    _ = orig_hp(b, embed=True)
                torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            avg = np.mean(times)
            orig_tp[bs] = bs / avg
            print(f"     batch={bs}: {avg*1000:.1f} ms, {bs/avg:.1f} cells/sec")
        except RuntimeError:
            print(f"     batch={bs}: OOM"); torch.cuda.empty_cache()

    # Reload TurboGene
    del orig_hp, orig; reset()
    print("\n   TurboGene model:")
    orig2, qmodel2, _, _ = load_all_models(args.checkpoint_dir)
    del orig2; reset()
    q_hp = qmodel2.half()

    turbo_tp = {}
    for bs in [1, 2, 4, 8, 12, 16]:
        try:
            reset()
            for _ in range(2):
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    _ = q_hp.forward_memory_efficient(g[:bs].cuda(), c[:bs].cuda(), embed=True)
            torch.cuda.synchronize()
            times = []
            for _ in range(5):
                gi, ci = g[:bs].cuda(), c[:bs].cuda()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    _ = q_hp.forward_memory_efficient(gi, ci, embed=True)
                torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            avg = np.mean(times)
            turbo_tp[bs] = bs / avg
            print(f"     batch={bs:2d}: {avg*1000:.1f} ms, {bs/avg:.1f} cells/sec")
        except RuntimeError:
            print(f"     batch={bs:2d}: OOM"); torch.cuda.empty_cache()

    # ── Quantization Overhead ──
    print("\n" + "=" * 70)
    print("6. Quantization vs Chunking Overhead (batch=4)")
    print("=" * 70)

    try:
        reset()
        bs_test = 4
        gi_t, ci_t = g[:bs_test].cuda(), c[:bs_test].cuda()

        # Warmup
        for _ in range(2):
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _ = q_hp.forward(gene_token_indices=gi_t, gene_counts=ci_t, embed=True)

        # (a) Standard forward (no chunking, no quantization)
        torch.cuda.synchronize()
        times_std = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _ = q_hp.forward(gene_token_indices=gi_t, gene_counts=ci_t, embed=True)
            torch.cuda.synchronize(); times_std.append(time.perf_counter() - t0)
        avg_std = np.mean(times_std)

        # (b) Quantized forward (chunking + quantization)
        times_quant = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _ = q_hp.forward_memory_efficient(gi_t, ci_t, embed=True)
            torch.cuda.synchronize(); times_quant.append(time.perf_counter() - t0)
        avg_quant = np.mean(times_quant)

        # (c) Decode-quality forward (chunking + quantization, same as memory_efficient)
        times_decode = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _ = q_hp.forward_with_decode_quantization(gi_t, ci_t, embed=True)
            torch.cuda.synchronize(); times_decode.append(time.perf_counter() - t0)
        avg_decode = np.mean(times_decode)

        print(f"   Standard (no quant):        {avg_std*1000:.1f} ms")
        print(f"   Memory-efficient (chunk+q):  {avg_quant*1000:.1f} ms")
        print(f"   Decode-quality (chunk+q):    {avg_decode*1000:.1f} ms")
        print(f"   Quantization overhead:       {(avg_quant/avg_std - 1)*100:.1f}%")
    except RuntimeError:
        print("   OOM at batch=4"); torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n   Classification:")
    for (ds, mn), m in cls_results.items():
        print(f"     {ds:6s} {mn:28s}: Acc={m['acc']:.4f} F1={m['f1']:.4f} ARI={m['ari']:.4f}")

    # Compute accuracy delta
    for ds in datasets:
        orig_key = (ds, "Original (batch=4)")
        turbo_key = (ds, "TurboGene (batch=16)")
        if orig_key in cls_results and turbo_key in cls_results:
            delta_acc = cls_results[turbo_key]["acc"] - cls_results[orig_key]["acc"]
            delta_f1 = cls_results[turbo_key]["f1"] - cls_results[orig_key]["f1"]
            print(f"     {ds:6s} Delta: Acc={delta_acc:+.4f} F1={delta_f1:+.4f}")

    print("\n   Embeddings:")
    for ds, s in emb_sims.items():
        print(f"     {ds}: cos_sim={s['mean']:.6f}")

    ob = max(orig_tp.values()) if orig_tp else 0
    tb = max(turbo_tp.values()) if turbo_tp else 0
    print(f"\n   Throughput:")
    print(f"     Original best:  {ob:.1f} cells/sec")
    print(f"     TurboGene best: {tb:.1f} cells/sec")
    if ob > 0:
        print(f"     Speedup:        {tb/ob:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
