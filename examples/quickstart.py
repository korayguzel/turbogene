"""
TurboGene Quickstart: Efficient Single-Cell Analysis on Consumer GPUs

Usage:
    python examples/quickstart.py --data your_data.h5ad --checkpoint-dir checkpoints/tf_sapiens
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_turbogene(checkpoint_dir, device="cuda"):
    """Load TurboGene-optimized TranscriptFormer."""
    from scripts.test_sdpa_e2e import load_original_model
    from turbogene.sdpa_model import SDPATranscriptformer
    from turbogene.quantizer import TurboGeneQuantizer
    from turbogene.quantized_model import QuantizedTranscriptformer

    orig, vocab = load_original_model(checkpoint_dir)
    orig = orig.to(device, torch.float32).eval()
    sdpa = SDPATranscriptformer(orig).to(device, torch.float32).eval()
    mc = orig.model_config
    nh, hd = mc.num_heads, mc.embed_dim // mc.num_heads
    sl = mc.seq_len + mc.aux_len

    quantizer = TurboGeneQuantizer(mc.num_layers, nh, hd, n_bits=3).to(device)
    k_a, v_a = [], []
    def mh(li):
        def hook(mod, args):
            inp = args[0]; B = inp.size(0)
            k_a.append((li, mod.linears[1](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
            v_a.append((li, mod.linears[2](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
        return hook
    hooks = [l.self_attn.register_forward_pre_hook(mh(i)) for i, l in enumerate(sdpa.transformer_encoder.encoder_layers)]
    gen = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        _ = sdpa(gene_token_indices=torch.randint(7, len(vocab), (2, sl), generator=gen).to(device),
                 gene_counts=torch.randint(1, 30, (2, sl), generator=gen).float().to(device))
    for h in hooks: h.remove()
    quantizer.calibrate(k_a, v_a)

    model = QuantizedTranscriptformer(sdpa, quantizer).to(device).eval()
    return model, vocab, mc


def process_adata(model, adata, vocab, mc, batch_size=16, device="cuda"):
    """Extract cell embeddings from AnnData with TurboGene."""
    max_len = mc.seq_len + mc.aux_len
    gene_col = "ensembl_id"
    if gene_col not in adata.var.columns:
        for c in ["feature_id"]:
            if c in adata.var.columns:
                adata.var["ensembl_id"] = adata.var[c]; break

    gene_ids = adata.var[gene_col].values
    vmap = {g: vocab[g] for g in gene_ids if g in vocab}
    pad_id = vocab["[PAD]"]
    X = adata.raw.X if adata.raw is not None else adata.X

    all_g, all_c, valid_idx = [], [], []
    for ci in range(X.shape[0]):
        row = X[ci].toarray().flatten() if hasattr(X[ci], "toarray") else np.asarray(X[ci]).flatten()
        nz = row > 0
        inv = [i for i, g in enumerate(gene_ids[nz]) if g in vmap]
        if len(inv) < 10: continue
        gi = np.array([vmap[gene_ids[nz][i]] for i in inv])
        cv = np.clip(row[nz][inv].astype(np.float32), 0, 30)
        if len(gi) > max_len:
            sel = np.random.choice(len(gi), max_len, replace=False)
            gi, cv = gi[sel], cv[sel]
        pad_n = max_len - len(gi)
        if pad_n > 0:
            gi = np.concatenate([gi, np.full(pad_n, pad_id, dtype=np.int64)])
            cv = np.concatenate([cv, np.zeros(pad_n, dtype=np.float32)])
        all_g.append(gi); all_c.append(cv); valid_idx.append(ci)

    g_t = torch.tensor(np.stack(all_g), dtype=torch.long)
    c_t = torch.tensor(np.stack(all_c), dtype=torch.float32)

    m = model.half()
    embs = []
    t0 = time.perf_counter()
    for i in range(0, len(all_g), batch_size):
        gi = g_t[i:i+batch_size].to(device)
        ci = c_t[i:i+batch_size].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = m.forward_memory_efficient(gi, ci, embed=True)
        embs.append(out["embeddings"].cpu())
    elapsed = time.perf_counter() - t0
    embs = torch.cat(embs, dim=0).numpy()
    print(f"Processed {len(all_g)} cells in {elapsed:.1f}s ({len(all_g)/elapsed:.1f} cells/sec)")

    result = adata[valid_idx].copy()
    result.obsm["turbogene_embeddings"] = embs
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="transcriptformer/test/data/human_val.h5ad")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/tf_sapiens")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default="output_embeddings.h5ad")
    args = parser.parse_args()

    print("Loading TurboGene model...")
    model, vocab, mc = load_turbogene(args.checkpoint_dir)

    print(f"Loading {args.data}...")
    adata = ad.read_h5ad(args.data)
    print(f"{adata.shape[0]} cells, {adata.shape[1]} genes")

    result = process_adata(model, adata, vocab, mc, batch_size=args.batch_size)
    result.write_h5ad(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
