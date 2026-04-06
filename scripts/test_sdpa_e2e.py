"""
End-to-end test: Original FlexAttention model vs SDPA-converted model.

Tests:
1. Log-likelihood equivalence (ZTP NLL + Gene ID CE)
2. Cell embedding cosine similarity
3. VRAM profiling at batch=1,4,8
"""

import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_gpu_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024


def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_original_model(checkpoint_dir):
    """Load original TranscriptFormer with FlexAttention."""
    from transcriptformer.tokenizer.vocab import construct_gene_embeddings
    from transcriptformer.model.model import Transcriptformer
    from transcriptformer.data.dataclasses import ModelConfig, DataConfig, LossConfig

    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    mc = config["model"]["model_config"]
    special_tokens = config["model"]["data_config"]["special_tokens"]
    emb_files = [str(checkpoint_dir / "vocabs" / f)
                 for f in config["model"]["data_config"]["esm2_mappings"]]

    gene_vocab_dict, emb_matrix = construct_gene_embeddings(emb_files, special_tokens)
    emb_matrix = torch.tensor(emb_matrix)

    # Aux vocab
    aux_vocab_path = checkpoint_dir / "vocabs" / "assay_vocab.json"
    aux_vocab_dict = None
    if aux_vocab_path.exists():
        with open(aux_vocab_path) as f:
            aux_vocab_dict = {"assay": json.load(f)}

    data_config = DataConfig(
        aux_vocab_path="", pin_memory=False, aux_cols=None,
        gene_col_name="ensembl_id", clip_counts=30, filter_to_vocabs=True,
        filter_outliers=0.0, pad_zeros=True, normalize_to_scale=0,
        n_data_workers=0, sort_genes=False, randomize_genes=False,
        min_expressed_genes=0, gene_pad_token="[PAD]", aux_pad_token="unknown",
    )

    model_config = ModelConfig(
        log_counts_eps=mc.get("log_counts_eps", 1e-6),
        num_heads=mc["num_heads"], num_layers=mc["num_layers"],
        model_dim=mc["model_dim"], embed_dim=mc["embed_dim"],
        dropout=0.0,  # Disable dropout for deterministic comparison
        activation=mc["activation"], attn_bias=mc["attn_bias"],
        fw_bias=mc["fw_bias"], mu_link_fn=mc["mu_link_fn"],
        softcap=mc["softcap"], seq_len=mc["seq_len"],
        aux_len=mc["aux_len"], block_len=mc["block_len"],
        gene_head_hidden_dim=mc.get("gene_head_hidden_dim", 2048),
        compile_block_mask=False,
    )

    loss_config = LossConfig(
        gene_id_loss_weight=mc.get("gene_id_loss_weight",
                                   config["model"]["loss_config"]["gene_id_loss_weight"]),
    )

    model = Transcriptformer(
        data_config=data_config, model_config=model_config,
        loss_config=loss_config, gene_vocab_dict=gene_vocab_dict,
        aux_vocab_dict=aux_vocab_dict, emb_matrix=emb_matrix,
    )

    weights_path = checkpoint_dir / "model_weights.pt"
    state_dict = torch.load(weights_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict

    return model, gene_vocab_dict


def create_test_batch(batch_size, seq_len, vocab_size, device):
    """Create deterministic test batch."""
    from transcriptformer.data.dataclasses import BatchData

    gen = torch.Generator(device="cpu").manual_seed(42)
    gene_ids = torch.randint(7, vocab_size, (batch_size, seq_len), generator=gen)
    counts = torch.randint(1, 30, (batch_size, seq_len), generator=gen).float()

    # 10% padding at end
    pad_start = seq_len - seq_len // 10
    gene_ids[:, pad_start:] = 5  # PAD idx
    counts[:, pad_start:] = 0

    return BatchData(
        gene_token_indices=gene_ids.to(device),
        gene_counts=counts.to(device),
        aux_token_indices=None,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float32  # FP32 for precise comparison

    print("=" * 60)
    print("End-to-End: FlexAttention vs SDPA Model")
    print("=" * 60)

    # ── 1. Load original model ──
    print("\n1. Loading original TranscriptFormer...")
    orig_model, gene_vocab_dict = load_original_model(args.checkpoint_dir)
    orig_model = orig_model.to(device, dtype).requires_grad_(False)
    vocab_size = len(gene_vocab_dict)
    mc = orig_model.model_config
    print(f"   Loaded: {mc.num_layers}L, {mc.num_heads}H, "
          f"embed_dim={mc.embed_dim}, vocab={vocab_size}")

    # ── 2. Convert to SDPA ──
    print("\n2. Converting to SDPA model...")
    from turbogene.sdpa_model import SDPATranscriptformer
    sdpa_model = SDPATranscriptformer(orig_model).to(device, dtype).requires_grad_(False)
    print("   Conversion complete.")

    # ── 3. Equivalence test ──
    seq_len = mc.seq_len + mc.aux_len  # 2048
    test_sizes = [1, 4]

    for bs in test_sizes:
        print(f"\n{'='*60}")
        print(f"3. Equivalence test: batch_size={bs}, seq_len={seq_len}")
        print(f"{'='*60}")

        batch = create_test_batch(bs, seq_len, vocab_size, device)

        # Original FlexAttention forward
        print("   Running original model...")
        with torch.no_grad():
            orig_output = orig_model(batch, embed=True)

        # SDPA forward
        print("   Running SDPA model...")
        with torch.no_grad():
            sdpa_output = sdpa_model(
                gene_token_indices=batch.gene_token_indices,
                gene_counts=batch.gene_counts,
                embed=True,
            )

        # Compare mu (count predictions)
        mu_cos = F.cosine_similarity(
            orig_output["mu"].reshape(bs, -1),
            sdpa_output["mu"].reshape(bs, -1),
            dim=-1,
        ).mean()
        mu_max_diff = (orig_output["mu"] - sdpa_output["mu"]).abs().max()

        print(f"\n   Count prediction (mu):")
        print(f"     Cosine similarity: {mu_cos.item():.6f}")
        print(f"     Max abs diff:      {mu_max_diff.item():.6e}")

        # Compare gene logits
        if "gene_logit" in orig_output and "gene_logit" in sdpa_output:
            gl_cos = F.cosine_similarity(
                orig_output["gene_logit"].reshape(bs, -1),
                sdpa_output["gene_logit"].reshape(bs, -1),
                dim=-1,
            ).mean()
            gl_max_diff = (orig_output["gene_logit"] - sdpa_output["gene_logit"]).abs().max()
            print(f"\n   Gene ID logits:")
            print(f"     Cosine similarity: {gl_cos.item():.6f}")
            print(f"     Max abs diff:      {gl_max_diff.item():.6e}")

        # Compare embeddings
        if "embeddings" in orig_output and "embeddings" in sdpa_output:
            emb_cos = F.cosine_similarity(
                orig_output["embeddings"].reshape(bs, -1),
                sdpa_output["embeddings"].reshape(bs, -1),
                dim=-1,
            ).mean()
            print(f"\n   Cell embeddings:")
            print(f"     Cosine similarity: {emb_cos.item():.6f}")

        # Compute log-likelihood
        orig_llh = orig_model.criterion(
            mu=orig_output["mu"], input_counts=orig_output["input_counts"],
            mask=orig_output["mask"], eval_mode=True,
        )
        sdpa_llh = sdpa_model.criterion(
            mu=sdpa_output["mu"], input_counts=sdpa_output["input_counts"],
            mask=sdpa_output["mask"], eval_mode=True,
        )
        llh_diff = (orig_llh - sdpa_llh).abs().mean()
        llh_rel = (llh_diff / orig_llh.abs().mean())
        print(f"\n   Log-likelihood (ZTP NLL):")
        print(f"     Original mean:  {orig_llh.mean().item():.4f}")
        print(f"     SDPA mean:      {sdpa_llh.mean().item():.4f}")
        print(f"     Abs diff:       {llh_diff.item():.6e}")
        print(f"     Relative diff:  {llh_rel.item():.6e}")

        passed = mu_cos.item() > 0.999
        print(f"\n   {'PASS' if passed else 'FAIL'}: mu cosine sim "
              f"{'>' if passed else '<='} 0.999")

        del batch, orig_output, sdpa_output
        reset_gpu()

    # ── 4. VRAM profiling ──
    print(f"\n{'='*60}")
    print("4. VRAM Profiling: SDPA model")
    print(f"{'='*60}")

    # Free original model
    del orig_model
    reset_gpu()

    sdpa_model = sdpa_model.half()  # Switch to FP16 for VRAM test
    reset_gpu()

    weight_mem = get_gpu_mb()
    print(f"\n   Model weights (FP16): {weight_mem:.1f} MB")

    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    for bs in [1, 4, 8, 16]:
        reset_gpu()
        base_mem = get_gpu_mb()

        try:
            gen = torch.Generator(device="cpu").manual_seed(42)
            gene_ids = torch.randint(7, vocab_size, (bs, seq_len), generator=gen).cuda()
            counts = torch.randint(1, 30, (bs, seq_len), generator=gen).float().cuda()
            pad_start = seq_len - seq_len // 10
            gene_ids[:, pad_start:] = 5
            counts[:, pad_start:] = 0

            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad(), torch.amp.autocast("cuda"):
                result = sdpa_model(gene_ids, counts)

            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            fwd = peak - base_mem

            # Theoretical KV cache
            kv_mb = (2 * mc.num_layers * mc.num_heads *
                     (mc.embed_dim // mc.num_heads) * seq_len * bs * 2) / 1024 / 1024

            print(f"\n   batch={bs:2d}: peak={peak:.0f} MB ({peak/gpu_total*100:.1f}% GPU), "
                  f"forward={fwd:.0f} MB, KV_cache={kv_mb:.0f} MB "
                  f"({kv_mb/peak*100:.1f}% of peak)")

            del result, gene_ids, counts

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n   batch={bs:2d}: OOM!")
                torch.cuda.empty_cache()
            else:
                raise

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
