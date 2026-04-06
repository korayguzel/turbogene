"""Download benchmark datasets for TurboGene large-scale validation.

Dataset 1: Tabula Sapiens subset — diverse human cell atlas (~50K cells)
Dataset 2: COVID-19 PBMC — disease state identification (~20K cells)

Uses CellxGene Census API for fast, targeted downloads.
"""

import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))

DATA_DIR = Path("data/benchmark")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_tabula_sapiens():
    """Download Tabula Sapiens subset via CellxGene Census."""
    out_path = DATA_DIR / "tabula_sapiens_50k.h5ad"
    if out_path.exists():
        adata = ad.read_h5ad(out_path)
        print(f"  Already exists: {adata.shape}")
        return out_path

    print("  Downloading Tabula Sapiens via CellxGene Census...")
    import cellxgene_census

    with cellxgene_census.open_soma() as census:
        # Query Tabula Sapiens (collection_id for TS)
        # Use a broad human query with tissue diversity
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=(
                "dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc'"  # Tabula Sapiens
            ),
            obs_column_names=[
                "cell_type", "tissue", "disease", "sex", "assay",
                "donor_id", "suspension_type",
            ],
            var_column_names=["feature_id", "feature_name"],
        )

    print(f"  Raw: {adata.shape}")

    # Subsample to 50K if larger
    if adata.n_obs > 50000:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, 50000, replace=False)
        adata = adata[idx].copy()

    # Ensure ensembl_id column
    if "feature_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["feature_id"]

    adata.write_h5ad(out_path)
    print(f"  Saved: {adata.shape} to {out_path}")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"  Tissues: {adata.obs['tissue'].nunique()}")
    return out_path


def download_covid_pbmc():
    """Download COVID-19 PBMC dataset via CellxGene Census."""
    out_path = DATA_DIR / "covid_pbmc_20k.h5ad"
    if out_path.exists():
        adata = ad.read_h5ad(out_path)
        print(f"  Already exists: {adata.shape}")
        return out_path

    print("  Downloading COVID-19 PBMC via CellxGene Census...")
    import cellxgene_census

    with cellxgene_census.open_soma() as census:
        # COVID-19 PBMC datasets
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=(
                "tissue_general == 'blood' and "
                "disease in ['COVID-19', 'normal']"
            ),
            obs_column_names=[
                "cell_type", "tissue", "disease", "sex", "assay",
                "donor_id", "suspension_type",
            ],
            var_column_names=["feature_id", "feature_name"],
        )

    print(f"  Raw: {adata.shape}")

    # Subsample to 20K — balanced across disease states
    target_per_class = 10000
    subsets = []
    for disease in adata.obs["disease"].unique():
        mask = adata.obs["disease"] == disease
        disease_adata = adata[mask]
        if disease_adata.n_obs > target_per_class:
            np.random.seed(42)
            idx = np.random.choice(disease_adata.n_obs, target_per_class, replace=False)
            subsets.append(disease_adata[idx].copy())
        else:
            subsets.append(disease_adata.copy())

    adata = ad.concat(subsets)

    if "feature_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["feature_id"]

    adata.write_h5ad(out_path)
    print(f"  Saved: {adata.shape} to {out_path}")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"  Disease: {adata.obs['disease'].value_counts().to_dict()}")
    return out_path


def download_large_atlas():
    """Download a large human atlas subset (100K cells) for case study."""
    out_path = DATA_DIR / "human_atlas_100k.h5ad"
    if out_path.exists():
        adata = ad.read_h5ad(out_path)
        print(f"  Already exists: {adata.shape}")
        return out_path

    print("  Downloading 100K human cells via CellxGene Census...")
    import cellxgene_census

    with cellxgene_census.open_soma() as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=(
                "tissue_general in ['blood', 'lung', 'heart', 'brain', 'liver'] and "
                "disease == 'normal'"
            ),
            obs_column_names=[
                "cell_type", "tissue", "tissue_general", "disease",
                "assay", "donor_id",
            ],
            var_column_names=["feature_id", "feature_name"],
        )

    print(f"  Raw: {adata.shape}")

    # Subsample to 100K with tissue balance
    target = 100000
    if adata.n_obs > target:
        np.random.seed(42)
        # Stratified by tissue_general
        subsets = []
        tissues = adata.obs["tissue_general"].unique()
        per_tissue = target // len(tissues)
        for tissue in tissues:
            mask = adata.obs["tissue_general"] == tissue
            t_adata = adata[mask]
            n = min(t_adata.n_obs, per_tissue)
            idx = np.random.choice(t_adata.n_obs, n, replace=False)
            subsets.append(t_adata[idx].copy())
        adata = ad.concat(subsets)

    if "feature_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["feature_id"]

    adata.write_h5ad(out_path)
    print(f"  Saved: {adata.shape} to {out_path}")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"  Tissues: {adata.obs['tissue_general'].value_counts().to_dict()}")
    return out_path


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading benchmark datasets")
    print("=" * 60)

    print("\n1. Tabula Sapiens (50K cells):")
    try:
        download_tabula_sapiens()
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n2. COVID-19 PBMC (20K cells):")
    try:
        download_covid_pbmc()
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n3. Human atlas (100K cells):")
    try:
        download_large_atlas()
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\nDone.")
