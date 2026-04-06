"""
TurboGene Phase 2 PoC: FlexAttention to SDPA + KV Cache Conversion

Demonstrates that TranscriptFormer's FlexAttention layer can be replaced with
manual attention (with softcap) while preserving output equivalence.
Also validates KV cache for incremental decode.

Usage:
    python scripts/poc_sdpa_conversion.py --checkpoint-dir ./checkpoints/tf_sapiens
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))


def build_attn_bias(log_counts, softcap, causal=True):
    """
    Pre-compute additive attention bias matrix replicating score_mod.

    Original score_mod: bias = log_counts[b, kv_idx] * (q_idx > kv_idx)
    Causal mask: q_idx >= kv_idx (positions on and below diagonal allowed)

    Returns: (batch, 1, seq_len, seq_len) float tensor with -inf for masked positions.
    """
    batch_size, seq_len = log_counts.shape
    device = log_counts.device
    dtype = log_counts.dtype

    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    causal_allowed = q_idx >= kv_idx       # (S, S)
    strictly_after = q_idx > kv_idx        # (S, S)

    # log_counts bias: only where q_idx > kv_idx
    counts_bias = log_counts.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)
    counts_bias = counts_bias * strictly_after.unsqueeze(0).unsqueeze(0).to(dtype)

    # Start with counts bias, mask non-causal with -inf
    attn_bias = counts_bias.clone()
    if causal:
        attn_bias = attn_bias.masked_fill(
            ~causal_allowed.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    return attn_bias


def manual_attention_with_softcap(q, k, v, attn_bias, softcap, dropout_p=0.0):
    """
    Manual scaled dot-product attention with softcap and additive bias.

    Replicates: score = QK^T/sqrt(d) + bias; score = tanh(score/cap)*cap; softmax; @V
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores + attn_bias

    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap

    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = F.dropout(weights, p=dropout_p)

    return torch.matmul(weights, v)


class MultiHeadSelfAttentionSDPA(nn.Module):
    """Drop-in replacement for MultiHeadSelfFlexAttn with KV cache support."""

    def __init__(self, d_model, nheads, bias=False):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.h = nheads
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=bias) for _ in range(4)
        ])

    def forward(self, inp, attn_bias=None, softcap=0, past_key_value=None, use_cache=False):
        batch_size = inp.size(0)

        q = self.linears[0](inp).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linears[1](inp).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linears[2](inp).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        if softcap > 0:
            o = manual_attention_with_softcap(q, k, v, attn_bias, softcap)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)

        o = o.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        output = self.linears[3](o)
        return output, present_key_value


def test_single_layer_equivalence(checkpoint_dir_str):
    """Test SDPA layer produces identical output to FlexAttention layer."""
    from transcriptformer.model.layers import MultiHeadSelfFlexAttn
    from transcriptformer.model.losses import logit_softcap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print("=" * 60)
    print("PoC: FlexAttention -> SDPA Equivalence Test")
    print("=" * 60)

    # Load config and weights
    print("\n1. Loading TF-Sapiens layer 0 weights...")
    checkpoint_dir = Path(checkpoint_dir_str)

    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    mc = config["model"]["model_config"]
    embed_dim = mc["embed_dim"]
    num_heads = mc["num_heads"]
    softcap = mc["softcap"]
    head_dim = embed_dim // num_heads

    print(f"   embed_dim={embed_dim}, num_heads={num_heads}, "
          f"head_dim={head_dim}, softcap={softcap}")

    state_dict = torch.load(
        checkpoint_dir / "model_weights.pt", weights_only=True, map_location="cpu",
    )

    layer0_prefix = "transformer_encoder.encoder_layers.0.self_attn."
    attn_weights = {
        k.replace(layer0_prefix, ""): v
        for k, v in state_dict.items()
        if k.startswith(layer0_prefix)
    }
    del state_dict
    print(f"   Keys: {list(attn_weights.keys())}")

    # Create both layers with same weights
    print("\n2. Creating FlexAttention and SDPA layers...")

    flex_attn_layer = MultiHeadSelfFlexAttn(
        d_model=embed_dim, nheads=num_heads, bias=False, compile_attention=False,
    ).to(device, dtype)
    flex_attn_layer.load_state_dict(attn_weights)
    flex_attn_layer.requires_grad_(False)

    sdpa_layer = MultiHeadSelfAttentionSDPA(
        d_model=embed_dim, nheads=num_heads, bias=False,
    ).to(device, dtype)
    sdpa_layer.load_state_dict(attn_weights)
    sdpa_layer.requires_grad_(False)

    # Synthetic test data
    seq_len = 256
    batch_size = 2

    print(f"\n3. Testing with batch={batch_size}, seq_len={seq_len}...")

    torch.manual_seed(42)
    inp = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    counts = torch.randint(1, 30, (batch_size, seq_len), device=device).float()
    log_counts = torch.log1p(counts + 1e-6)

    # ── FlexAttention forward ──
    print("\n4. Running FlexAttention forward...")

    from torch.nn.attention.flex_attention import (
        flex_attention as fa_func,
        create_block_mask,
        and_masks,
    )

    def causal_mask_fn(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def pad_mask_factory(mask):
        def pad_mask(b, h, q_idx, kv_idx):
            return mask[b, kv_idx]
        return pad_mask

    pad_mask_bool = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
    mask_mod = and_masks(pad_mask_factory(pad_mask_bool), causal_mask_fn)

    block_mask = create_block_mask(
        mask_mod, batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device, BLOCK_SIZE=128, _compile=False,
    )

    lc_captured = log_counts
    sc_captured = softcap

    def score_mod_fn(score, b, h, q_idx, kv_idx):
        bias = lc_captured[b, kv_idx] * (q_idx > kv_idx)
        score = score + bias
        if sc_captured > 0:
            score = logit_softcap(score, sc_captured)
        return score

    with torch.no_grad():
        flex_output = flex_attn_layer(inp, score_mod=score_mod_fn, block_mask=block_mask)

    # ── SDPA forward ──
    print("5. Running SDPA forward...")

    attn_bias = build_attn_bias(log_counts, softcap, causal=True)

    with torch.no_grad():
        sdpa_output, _ = sdpa_layer(inp, attn_bias=attn_bias, softcap=softcap)

    # ── Compare ──
    print("\n6. Comparing outputs...")

    cos_sim = F.cosine_similarity(
        flex_output.reshape(batch_size, -1),
        sdpa_output.reshape(batch_size, -1),
        dim=-1,
    )
    max_abs_diff = (flex_output - sdpa_output).abs().max()
    rel_error = ((flex_output - sdpa_output).abs() / (flex_output.abs() + 1e-8)).mean()

    print(f"   Cosine similarity:  {cos_sim.mean().item():.6f} (target: >0.999)")
    print(f"   Max abs difference: {max_abs_diff.item():.6e}")
    print(f"   Relative error:     {rel_error.item():.6e}")

    passed = cos_sim.mean().item() > 0.999
    status = "PASS" if passed else "FAIL"
    print(f"\n   {status}: Cosine similarity {'>' if passed else '<='} 0.999")

    # ── KV Cache Test ──
    print("\n" + "=" * 60)
    print("7. Testing KV Cache (incremental decode)...")
    print("=" * 60)

    full_seq_len = 128
    inp_full = torch.randn(1, full_seq_len, embed_dim, device=device, dtype=dtype)
    lc_full = torch.log1p(
        torch.randint(1, 30, (1, full_seq_len), device=device).float() + 1e-6
    )

    attn_bias_full = build_attn_bias(lc_full, softcap, causal=True)

    with torch.no_grad():
        full_output, full_kv = sdpa_layer(
            inp_full, attn_bias=attn_bias_full, softcap=softcap, use_cache=True
        )

    # Incremental: prefill first 100, then decode 28 tokens one by one
    prefix_len = 100
    inp_prefix = inp_full[:, :prefix_len, :]
    attn_bias_prefix = build_attn_bias(lc_full[:, :prefix_len], softcap, causal=True)

    with torch.no_grad():
        prefix_output, kv_cache = sdpa_layer(
            inp_prefix, attn_bias=attn_bias_prefix, softcap=softcap, use_cache=True
        )

    decode_outputs = [prefix_output]
    for t in range(prefix_len, full_seq_len):
        inp_t = inp_full[:, t:t+1, :]
        kv_len = t + 1

        # Build single-query bias: (1, 1, 1, kv_len)
        bias_decode = lc_full[:, :kv_len].unsqueeze(1).unsqueeze(2)
        # Zero out last position (diagonal: q_idx == kv_idx, not strictly after)
        strictly_mask = torch.ones(1, 1, 1, kv_len, device=device, dtype=dtype)
        strictly_mask[0, 0, 0, -1] = 0.0
        bias_decode = bias_decode * strictly_mask

        with torch.no_grad():
            decode_out, kv_cache = sdpa_layer(
                inp_t, attn_bias=bias_decode, softcap=softcap,
                past_key_value=kv_cache, use_cache=True,
            )
        decode_outputs.append(decode_out)

    incremental_output = torch.cat(decode_outputs, dim=1)

    kv_cos_sim = F.cosine_similarity(
        full_output.reshape(1, -1),
        incremental_output.reshape(1, -1),
        dim=-1,
    )
    kv_max_diff = (full_output - incremental_output).abs().max()

    print(f"   Full prefill vs incremental decode:")
    print(f"   Cosine similarity:  {kv_cos_sim.item():.6f}")
    print(f"   Max abs difference: {kv_max_diff.item():.6e}")

    kv_passed = kv_cos_sim.item() > 0.999
    print(f"\n   {'PASS' if kv_passed else 'FAIL'}: KV cache cosine similarity "
          f"{'>' if kv_passed else '<='} 0.999")

    # ── Memory comparison ──
    print("\n" + "=" * 60)
    print("8. Memory comparison (single layer, batch=2, seq=256)")
    print("=" * 60)

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = flex_attn_layer(inp, score_mod=score_mod_fn, block_mask=block_mask)
    flex_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _, _ = sdpa_layer(inp, attn_bias=attn_bias, softcap=softcap)
    sdpa_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"   FlexAttention peak: {flex_mem:.1f} MB")
    print(f"   SDPA peak:          {sdpa_mem:.1f} MB")
    print(f"   Difference:         {flex_mem - sdpa_mem:+.1f} MB")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FlexAttn -> SDPA equiv:  {'PASS' if passed else 'FAIL'} "
          f"(cos_sim={cos_sim.mean().item():.6f})")
    print(f"  KV cache correctness:    {'PASS' if kv_passed else 'FAIL'} "
          f"(cos_sim={kv_cos_sim.item():.6f})")
    print(f"  Memory savings:          {flex_mem - sdpa_mem:+.1f} MB per layer")

    return passed and kv_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()
    success = test_single_layer_equivalence(args.checkpoint_dir)
    sys.exit(0 if success else 1)
