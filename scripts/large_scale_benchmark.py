"""
TurboGene Large-Scale Benchmark

1. Cell type classification on large datasets
2. Case study: process 50K+ cells on RTX 4070
3. Timing comparison
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
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def load_models(checkpoint_dir):
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
    q = TurboGeneQuantizer(mc.num_layers, nh, hd, n_bits=3).cuda()

    k_a, v_a = [], []
    def mh(li):
        def hook(mod, args):
            inp = args[0]; B = inp.size(0)
            k_a.append((li, mod.linears[1](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
            v_a.append((li, mod.linears[2](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
        return hook
    hooks = [l.self_attn.register_forward_pre_hook(mh(i)) for i,l in enumerate(sdpa.transformer_encoder.encoder_layers)]
    gen = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        _ = sdpa(gene_token_indices=torch.randint(7, len(vocab), (2, sl), generator=gen).cuda(),
                 gene_counts=torch.randint(1, 30, (2, sl), generator=gen).float().cuda())
    for h in hooks: h.remove()
    q.calibrate(k_a, v_a); del k_a, v_a

    qm = QuantizedTranscriptformer(sdpa, q).to("cuda", torch.float32).eval()
    return orig, qm, vocab, mc


def prepare(path, vocab, max_len):
    adata = ad.read_h5ad(path)
    gc_name = "ensembl_id"
    if gc_name not in adata.var.columns:
        for c in ["feature_id", "gene_ids"]:
            if c in adata.var.columns:
                adata.var["ensembl_id"] = adata.var[c]; break

    if gc_name not in adata.var.columns:
        return None

    gene_ids = adata.var[gc_name].values
    vmap = {g: vocab[g] for g in gene_ids if g in vocab}
    pad_id = vocab["[PAD]"]

    label_col = None
    for c in ["cell_type", "cell_state"]:
        if c in adata.obs.columns:
            label_col = c; break

    X = adata.raw.X if adata.raw is not None else adata.X
    all_g, all_c, labels = [], [], []

    for ci in range(X.shape[0]):
        row = X[ci].toarray().flatten() if hasattr(X[ci], "toarray") else np.asarray(X[ci]).flatten()
        nz = row > 0
        genes_nz = gene_ids[nz]
        counts_nz = row[nz]
        inv = [i for i, g in enumerate(genes_nz) if g in vmap]
        if len(inv) < 10: continue

        gi = np.array([vmap[genes_nz[i]] for i in inv])
        cv = np.clip(counts_nz[inv].astype(np.float32), 0, 30)
        if len(gi) > max_len:
            sel = np.random.choice(len(gi), max_len, replace=False)
            gi, cv = gi[sel], cv[sel]
        pad_n = max_len - len(gi)
        if pad_n > 0:
            gi = np.concatenate([gi, np.full(pad_n, pad_id, dtype=np.int64)])
            cv = np.concatenate([cv, np.zeros(pad_n, dtype=np.float32)])
        all_g.append(gi); all_c.append(cv)
        if label_col: labels.append(adata.obs[label_col].iloc[ci])
        if (ci+1) % 10000 == 0: print(f"    {ci+1}/{X.shape[0]} cells...")

    if not all_g: return None
    return {"g": torch.tensor(np.stack(all_g), dtype=torch.long),
            "c": torch.tensor(np.stack(all_c), dtype=torch.float32),
            "labels": np.array(labels) if labels else None,
            "label_col": label_col, "n": len(all_g)}


def embed_timed(model, g, c, bs, eff=False, is_orig=False):
    from transcriptformer.data.dataclasses import BatchData
    N = g.size(0); embs = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(0, N, bs):
        gi, ci = g[i:i+bs].cuda(), c[i:i+bs].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            if eff:
                out = model.forward_memory_efficient(gi, ci, embed=True)
            elif is_orig:
                out = model(BatchData(gene_token_indices=gi, gene_counts=ci, aux_token_indices=None), embed=True)
            else:
                out = model(gene_token_indices=gi, gene_counts=ci, embed=True)
        embs.append(out["embeddings"].cpu()); del gi, ci
    torch.cuda.synchronize()
    return torch.cat(embs, dim=0).numpy(), time.perf_counter() - t0


def knn_cv(embs, labels, k=15, n_splits=5, max_n=20000):
    le = LabelEncoder(); y = le.fit_transform(labels)
    if len(y) > max_n:
        np.random.seed(42); idx = np.random.choice(len(y), max_n, replace=False)
        embs, y = embs[idx], y[idx]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, aris = [], [], []
    for tr, te in skf.split(embs, y):
        knn = KNeighborsClassifier(n_neighbors=min(k, len(tr)-1))
        knn.fit(embs[tr], y[tr]); pred = knn.predict(embs[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="weighted"))
        aris.append(adjusted_rand_score(y[te], pred))
    return {"acc": np.mean(accs), "f1": np.mean(f1s), "ari": np.mean(aris)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/benchmark")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\n1. Loading models...")
    orig, qm, vocab, mc = load_models(args.checkpoint_dir)
    sl = mc.seq_len + mc.aux_len

    # Find datasets
    ddir = Path(args.data_dir)
    datasets = {}
    for name, fname in [("human_10k", "human_10k.h5ad"),
                         ("human_50k", "human_50k.h5ad")]:
        p = ddir / fname
        if p.exists():
            print(f"\n2. Preparing {name}...")
            d = prepare(str(p), vocab, sl)
            if d: datasets[name] = d; print(f"   {d['n']} cells ready")

    # Fallback
    if not datasets:
        print("\n   Fallback to test data")
        d = prepare("transcriptformer/test/data/human_val.h5ad", vocab, sl)
        if d: datasets["human_val"] = d

    # Classification
    print("\n" + "=" * 70)
    print("3. Classification")
    print("=" * 70)
    for ds, data in datasets.items():
        if data["labels"] is None: continue
        nt = len(set(data["labels"]))
        print(f"\n   {ds}: {data['n']} cells, {nt} types")
        for mn, mdl, bs, eff, io in [("Original(b=4)", orig, 4, False, True),
                                      ("TurboGene(b=16)", qm, 16, True, False)]:
            reset(); m = mdl.half()
            embs, t = embed_timed(m, data["g"], data["c"], bs, eff, io)
            mdl.float()
            met = knn_cv(embs, data["labels"])
            cps = data["n"] / t
            print(f"     {mn:20s}: Acc={met['acc']:.4f} F1={met['f1']:.4f} ARI={met['ari']:.4f} | {t:.1f}s ({cps:.1f} c/s)")

    # Case study: largest dataset
    largest = max(datasets.items(), key=lambda x: x[1]["n"])
    ds_name, data = largest
    print(f"\n{'='*70}")
    print(f"4. Case Study: {data['n']} cells on RTX 4070")
    print(f"{'='*70}")

    reset(); orig_hp = orig.half()
    try:
        _, ot = embed_timed(orig_hp, data["g"], data["c"], 4, False, True)
        print(f"   Original(b=4):  {ot:.1f}s ({data['n']/ot:.1f} c/s)")
    except RuntimeError:
        ot = None; print("   Original(b=4):  OOM!"); torch.cuda.empty_cache()
    orig.float()

    reset(); q_hp = qm.half()
    _, tt = embed_timed(q_hp, data["g"], data["c"], 16, True, False)
    print(f"   TurboGene(b=16): {tt:.1f}s ({data['n']/tt:.1f} c/s)")
    qm.float()

    if ot: print(f"   Speedup: {ot/tt:.2f}x")
    print(f"   {data['n']} cells in {tt:.0f}s ({tt/60:.1f} min) on RTX 4070")

    # Embedding similarity
    print(f"\n{'='*70}")
    print("5. Embedding Similarity")
    print(f"{'='*70}")
    for ds, data in datasets.items():
        n = min(data["n"], 5000)
        reset()
        e1, _ = embed_timed(orig.half(), data["g"][:n], data["c"][:n], 4, False, True); orig.float()
        e2, _ = embed_timed(qm.half(), data["g"][:n], data["c"][:n], 16, True, False); qm.float()
        cos = F.cosine_similarity(torch.tensor(e1), torch.tensor(e2), dim=-1)
        print(f"   {ds} ({n} cells): cos={cos.mean():.6f} min={cos.min():.6f}")

    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
