"""
TurboGene Phase 1: TranscriptFormer VRAM Profiling & K/V Activation Analysis

This script:
1. Loads TF-Sapiens checkpoint
2. Profiles VRAM usage at batch=1,4,8,16 (model weights vs KV-equivalent vs activations)
3. Extracts K/V activation distributions (head-wise histograms + outlier analysis)
4. Saves results to docs/profiling_results/

Usage:
    python scripts/profile_transcriptformer.py --checkpoint-dir ./checkpoints/tf_sapiens
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add transcriptformer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
    return {"allocated": 0, "reserved": 0, "max_allocated": 0}


def reset_memory_stats():
    """Reset GPU memory statistics."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def profile_model_weights(model):
    """Profile model weight memory."""
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        size = param.numel()
        total_params += size
        if param.requires_grad:
            trainable_params += size
        else:
            frozen_params += size

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_mb_fp32": total_params * 4 / 1024 / 1024,
        "total_mb_fp16": total_params * 2 / 1024 / 1024,
        "trainable_mb_fp16": trainable_params * 2 / 1024 / 1024,
        "frozen_mb_fp16": frozen_params * 2 / 1024 / 1024,
    }


def compute_kv_cache_size(num_layers, num_heads, head_dim, seq_len, batch_size, dtype_bytes=2):
    """Compute theoretical KV cache size."""
    # 2 for K and V
    kv_size = 2 * num_layers * num_heads * head_dim * seq_len * batch_size * dtype_bytes
    return kv_size / 1024 / 1024  # MB


class KVActivationHook:
    """Hook to capture K and V activations from attention layers."""

    def __init__(self):
        self.k_activations = []  # List of (layer_idx, tensor)
        self.v_activations = []

    def create_hook(self, layer_idx):
        """Create a forward hook for a specific attention layer."""
        def hook_fn(module, input_args, output):
            # MultiHeadSelfFlexAttn: input is (inp,), Q/K/V computed inside
            # We need to capture the K and V after linear projection
            inp = input_args[0]
            batch_size = inp.size(0)
            h = module.h
            d_k = module.d_k

            # Recompute Q, K, V (same as forward pass)
            with torch.no_grad():
                k = module.linears[1](inp).view(batch_size, -1, h, d_k).transpose(1, 2)
                v = module.linears[2](inp).view(batch_size, -1, h, d_k).transpose(1, 2)

            self.k_activations.append((layer_idx, k.detach().cpu()))
            self.v_activations.append((layer_idx, v.detach().cpu()))

        return hook_fn


def analyze_kv_distributions(k_activations, v_activations, output_dir):
    """Analyze K/V activation distributions per head."""
    results = {"keys": {}, "values": {}}

    for kv_type, activations, result_key in [
        ("K", k_activations, "keys"),
        ("V", v_activations, "values"),
    ]:
        for layer_idx, tensor in activations:
            # tensor shape: (batch, num_heads, seq_len, head_dim)
            batch_size, num_heads, seq_len, head_dim = tensor.shape

            layer_stats = {}
            for head_idx in range(num_heads):
                head_data = tensor[:, head_idx, :, :].float()  # (batch, seq, head_dim)

                # Flatten across batch and seq for distribution analysis
                flat = head_data.reshape(-1)
                valid = flat[flat != 0]  # exclude padding zeros

                if len(valid) == 0:
                    continue

                stats = {
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "median": float(valid.median()),
                    "q01": float(valid.quantile(0.01)),
                    "q99": float(valid.quantile(0.99)),
                    "abs_max": float(valid.abs().max()),
                    "abs_mean": float(valid.abs().mean()),
                    "outlier_ratio": float((valid.abs() > 3 * valid.std()).float().mean()),
                    "kurtosis": float(((valid - valid.mean()) / valid.std()).pow(4).mean() - 3),
                }

                # Outlier-to-median ratio (like TurboESM analysis)
                median_abs = float(valid.abs().median())
                if median_abs > 0:
                    stats["outlier_to_median_ratio"] = float(valid.abs().max() / median_abs)
                else:
                    stats["outlier_to_median_ratio"] = float("inf")

                layer_stats[f"head_{head_idx}"] = stats

                # Save histogram data
                hist_values, bin_edges = np.histogram(
                    valid.numpy(), bins=200, density=True
                )
                np.savez(
                    output_dir / f"{kv_type}_layer{layer_idx}_head{head_idx}_hist.npz",
                    values=hist_values,
                    bin_edges=bin_edges,
                )

            results[result_key][f"layer_{layer_idx}"] = layer_stats

    return results


def create_synthetic_batch(batch_size, seq_len, vocab_size, device):
    """Create a synthetic batch for profiling."""
    from transcriptformer.data.dataclasses import BatchData

    gene_token_indices = torch.randint(7, vocab_size, (batch_size, seq_len), device=device)
    gene_counts = torch.randint(1, 30, (batch_size, seq_len), device=device).float()

    # Add some padding
    pad_start = seq_len - seq_len // 10  # 10% padding
    gene_token_indices[:, pad_start:] = 5  # PAD token index
    gene_counts[:, pad_start:] = 0

    return BatchData(
        gene_token_indices=gene_token_indices,
        gene_counts=gene_counts,
        aux_token_indices=None,
    )


def main():
    parser = argparse.ArgumentParser(description="TranscriptFormer VRAM Profiling")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Path to TF-Sapiens checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="docs/profiling_results",
                       help="Output directory for profiling results")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16",
                       help="Comma-separated batch sizes to profile")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("=" * 60)
    print("TurboGene Phase 1: TranscriptFormer VRAM Profiling")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"GPU: {gpu_name} ({gpu_total_mb:.0f} MB)")

    # Load model
    checkpoint_dir = Path(args.checkpoint_dir)
    print(f"\nLoading model from {checkpoint_dir}...")

    from transcriptformer.tokenizer.vocab import construct_gene_embeddings
    from transcriptformer.model.model import Transcriptformer
    from transcriptformer.data.dataclasses import ModelConfig, DataConfig, LossConfig

    # Load config from checkpoint
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    print(f"Config loaded.")

    # Load vocabs and embeddings
    vocabs_dir = checkpoint_dir / "vocabs"
    print(f"Loading vocabularies from {vocabs_dir}...")

    # Find embedding files
    import glob
    emb_files = glob.glob(str(vocabs_dir / "*.h5"))
    if not emb_files:
        emb_files = glob.glob(str(checkpoint_dir / "*.h5"))

    print(f"Found embedding files: {emb_files}")

    # Try to construct vocabs
    special_tokens = config.get("model", {}).get("data_config", {}).get("special_tokens",
        ["unknown", "[START]", "[END]", "[RD]", "[CELL]", "[PAD]", "[MASK]"])

    gene_vocab_dict, emb_matrix = construct_gene_embeddings(emb_files, special_tokens)
    emb_matrix = torch.tensor(emb_matrix)

    # Load aux vocab (assay)
    aux_vocab_path = vocabs_dir / "assay_vocab.json"
    aux_vocab_dict = None
    if aux_vocab_path.exists():
        with open(aux_vocab_path) as f:
            aux_vocab_dict = {"assay": json.load(f)}
        print(f"Aux vocabulary loaded: assay ({len(aux_vocab_dict['assay'])} entries)")

    print(f"Gene vocabulary size: {len(gene_vocab_dict)}")
    print(f"Embedding matrix shape: {emb_matrix.shape}")

    # Create model config
    mc = config.get("model", {}).get("model_config", {})
    model_config = ModelConfig(
        log_counts_eps=mc.get("log_counts_eps", 1e-6),
        num_heads=mc.get("num_heads", 16),
        num_layers=mc.get("num_layers", 12),
        model_dim=mc.get("model_dim", 2048),
        embed_dim=mc.get("embed_dim", 2048),
        dropout=mc.get("dropout", 0.0),
        activation=mc.get("activation", "gelu"),
        attn_bias=mc.get("attn_bias", False),
        fw_bias=mc.get("fw_bias", False),
        mu_link_fn=mc.get("mu_link_fn", "softplus"),
        softcap=mc.get("softcap", 10),
        seq_len=mc.get("seq_len", 2047),
        aux_len=mc.get("aux_len", 1),
        block_len=mc.get("block_len", 128),
        gene_head_hidden_dim=mc.get("gene_head_hidden_dim", 2048),
        compile_block_mask=False,  # Disable compilation for profiling
    )

    lc = config.get("model", {}).get("loss_config", {})
    loss_config = LossConfig(
        gene_id_loss_weight=lc.get("gene_id_loss_weight", 1.0),
    )

    # Create minimal data_config for loss initialization
    data_config = DataConfig(
        aux_vocab_path="",
        pin_memory=False,
        aux_cols=None,
        gene_col_name="ensembl_id",
        clip_counts=30,
        filter_to_vocabs=True,
        filter_outliers=0.0,
        pad_zeros=True,
        normalize_to_scale=0,
        n_data_workers=0,
        sort_genes=False,
        randomize_genes=False,
        min_expressed_genes=0,
        gene_pad_token="[PAD]",
        aux_pad_token="unknown",
    )

    # Instantiate model
    print(f"\nInstantiating model (embed_dim={model_config.embed_dim}, "
          f"heads={model_config.num_heads}, layers={model_config.num_layers})...")

    reset_memory_stats()
    mem_before = get_gpu_memory_mb()

    model = Transcriptformer(
        data_config=data_config,
        model_config=model_config,
        loss_config=loss_config,
        gene_vocab_dict=gene_vocab_dict,
        aux_vocab_dict=aux_vocab_dict,
        emb_matrix=emb_matrix,
    )

    # Load weights
    weights_path = checkpoint_dir / "model_weights.pt"
    if weights_path.exists():
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state_dict)
        del state_dict

    model = model.half().cuda()
    model.requires_grad_(False)

    mem_after_load = get_gpu_memory_mb()
    weight_mem = mem_after_load["allocated"] - mem_before["allocated"]

    # Profile model weights
    weight_stats = profile_model_weights(model)
    print(f"\n--- Model Weight Profile ---")
    print(f"Total params: {weight_stats['total_params']:,}")
    print(f"Trainable: {weight_stats['trainable_params']:,} ({weight_stats['trainable_mb_fp16']:.1f} MB FP16)")
    print(f"Frozen: {weight_stats['frozen_params']:,} ({weight_stats['frozen_mb_fp16']:.1f} MB FP16)")
    print(f"Actual GPU memory for weights: {weight_mem:.1f} MB")

    # Theoretical KV cache sizes
    print(f"\n--- Theoretical KV Cache Sizes (FP16) ---")
    num_layers = model_config.num_layers
    num_heads = model_config.num_heads
    head_dim = model_config.embed_dim // model_config.num_heads
    seq_len = model_config.seq_len + model_config.aux_len  # 2048

    kv_sizes = {}
    for bs in batch_sizes:
        kv_mb = compute_kv_cache_size(num_layers, num_heads, head_dim, seq_len, bs)
        kv_sizes[bs] = kv_mb
        print(f"  batch={bs:2d}: KV cache = {kv_mb:.1f} MB "
              f"({kv_mb / gpu_total_mb * 100:.1f}% of GPU)")

    # VRAM profiling with actual forward passes
    print(f"\n--- VRAM Profiling (Actual Forward Pass) ---")
    vram_results = {}
    vocab_size = len(gene_vocab_dict)

    for bs in batch_sizes:
        print(f"\n  batch_size={bs}:")
        reset_memory_stats()

        try:
            batch = create_synthetic_batch(bs, seq_len, vocab_size, "cuda")

            mem_before_fwd = get_gpu_memory_mb()

            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(batch)

            mem_after_fwd = get_gpu_memory_mb()
            peak_mem = mem_after_fwd["max_allocated"]
            fwd_mem = peak_mem - mem_before_fwd["allocated"]

            vram_results[bs] = {
                "weight_mem_mb": weight_mem,
                "forward_pass_mem_mb": fwd_mem,
                "peak_mem_mb": peak_mem,
                "theoretical_kv_cache_mb": kv_sizes[bs],
                "activation_estimate_mb": fwd_mem - kv_sizes[bs],
                "gpu_utilization_pct": peak_mem / gpu_total_mb * 100,
                "kv_cache_pct_of_total": kv_sizes[bs] / peak_mem * 100 if peak_mem > 0 else 0,
            }

            print(f"    Weights:     {weight_mem:.1f} MB")
            print(f"    Forward:     {fwd_mem:.1f} MB")
            print(f"    Peak total:  {peak_mem:.1f} MB ({peak_mem/gpu_total_mb*100:.1f}% GPU)")
            print(f"    KV cache:    {kv_sizes[bs]:.1f} MB ({kv_sizes[bs]/peak_mem*100:.1f}% of peak)")

            del output, batch

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    OOM! Cannot fit batch_size={bs}")
                vram_results[bs] = {"error": "OOM"}
                torch.cuda.empty_cache()
            else:
                raise

    # K/V Activation Analysis (batch=1 for detailed analysis)
    print(f"\n--- K/V Activation Distribution Analysis ---")
    reset_memory_stats()

    kv_hook = KVActivationHook()
    hooks = []

    for layer_idx, layer in enumerate(model.transformer_encoder.encoder_layers):
        hook = layer.self_attn.register_forward_hook(kv_hook.create_hook(layer_idx))
        hooks.append(hook)

    # Run a forward pass to capture activations
    batch = create_synthetic_batch(1, seq_len, vocab_size, "cuda")
    with torch.no_grad(), torch.cuda.amp.autocast():
        _ = model(batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze distributions
    kv_dist_results = analyze_kv_distributions(
        kv_hook.k_activations, kv_hook.v_activations, output_dir
    )

    # Print summary
    print(f"\n  Head-wise outlier-to-median ratios:")
    for kv_type in ["keys", "values"]:
        print(f"\n  {kv_type.upper()}:")
        for layer_name, layer_data in kv_dist_results[kv_type].items():
            ratios = [h["outlier_to_median_ratio"] for h in layer_data.values()
                     if h["outlier_to_median_ratio"] != float("inf")]
            if ratios:
                print(f"    {layer_name}: "
                      f"mean={np.mean(ratios):.1f}x, "
                      f"max={np.max(ratios):.1f}x, "
                      f"min={np.min(ratios):.1f}x")

    # Save all results
    all_results = {
        "gpu": {"name": gpu_name, "total_mb": gpu_total_mb},
        "model": {
            "variant": "TF-Sapiens",
            "embed_dim": model_config.embed_dim,
            "num_heads": model_config.num_heads,
            "num_layers": model_config.num_layers,
            "head_dim": head_dim,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
        },
        "weight_profile": weight_stats,
        "vram_profile": vram_results,
        "kv_cache_theoretical": {str(k): v for k, v in kv_sizes.items()},
        "kv_distributions": kv_dist_results,
    }

    results_path = output_dir / "profiling_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"Histogram data saved to {output_dir}/*.npz")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
