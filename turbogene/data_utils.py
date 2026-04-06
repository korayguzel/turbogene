"""Fast data preparation utilities for TurboGene."""

import numpy as np
import torch
from scipy.sparse import issparse


def fast_tokenize(adata, vocab, max_len, min_genes=10):
    """Vectorized tokenization of AnnData for TranscriptFormer inference.

    Much faster than per-cell loop for large sparse datasets.

    Returns:
        gene_ids: (N, max_len) int64 tensor
        counts: (N, max_len) float32 tensor
        valid_indices: list of valid cell indices
    """
    # Find ensembl_id column
    gene_col = None
    for c in ["ensembl_id", "feature_id"]:
        if c in adata.var.columns:
            gene_col = c
            break
    if gene_col is None and adata.var.index.name:
        gene_ids_arr = adata.var.index.values
    else:
        gene_ids_arr = adata.var[gene_col].values

    # Build vocab mapping (gene_idx_in_adata -> token_id)
    pad_id = vocab["[PAD]"]
    gene_to_tok = {}
    for i, g in enumerate(gene_ids_arr):
        if g in vocab:
            gene_to_tok[i] = vocab[g]

    valid_gene_indices = sorted(gene_to_tok.keys())
    tok_ids = np.array([gene_to_tok[i] for i in valid_gene_indices])

    X = adata.raw.X if adata.raw is not None else adata.X

    # Pre-slice columns ONCE then process rows
    if issparse(X):
        X_sub = X.tocsc()[:, valid_gene_indices].tocsr()  # CSC for col slice, CSR for row slice
    else:
        X_sub = X[:, valid_gene_indices]

    all_g = []
    all_c = []
    valid_cells = []

    n_cells = X_sub.shape[0]
    batch_size = 1000

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)

        if issparse(X_sub):
            block = X_sub[start:end].toarray()
        else:
            block = X_sub[start:end]

        for local_i in range(block.shape[0]):
            global_i = start + local_i
            row = block[local_i]
            nz_mask = row > 0
            n_nz = nz_mask.sum()

            if n_nz < min_genes:
                continue

            nz_toks = tok_ids[nz_mask]
            nz_counts = np.clip(row[nz_mask].astype(np.float32), 0, 30)

            # Truncate if too long
            if n_nz > max_len:
                sel = np.random.choice(n_nz, max_len, replace=False)
                nz_toks = nz_toks[sel]
                nz_counts = nz_counts[sel]
                n_nz = max_len

            # Pad
            pad_n = max_len - n_nz
            if pad_n > 0:
                nz_toks = np.concatenate([nz_toks, np.full(pad_n, pad_id, dtype=np.int64)])
                nz_counts = np.concatenate([nz_counts, np.zeros(pad_n, dtype=np.float32)])

            all_g.append(nz_toks)
            all_c.append(nz_counts)
            valid_cells.append(global_i)

        if (end) % 5000 == 0 or end == n_cells:
            print(f"  Tokenized {end}/{n_cells} cells ({len(valid_cells)} valid)")

    if not all_g:
        return None, None, []

    return (
        torch.tensor(np.stack(all_g), dtype=torch.long),
        torch.tensor(np.stack(all_c), dtype=torch.float32),
        valid_cells,
    )
