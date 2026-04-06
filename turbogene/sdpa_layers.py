"""
SDPA-based attention layers for TranscriptFormer.

Drop-in replacements for FlexAttention layers with:
- Softcap support via manual attention
- KV cache for incremental decode
- Pre-computed additive attention bias (replaces score_mod)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def build_attn_bias_and_mask(
    log_counts: Tensor,
    pad_mask: Tensor,
    causal: bool = True,
    emb_mode: bool = False,
) -> tuple:
    """
    Pre-compute additive attention bias AND boolean mask separately.

    We MUST keep these separate because softcap (tanh) would turn -inf into
    -softcap, leaking masked positions into attention. The correct order is:
    1. scores = QK^T/sqrt(d) + count_bias
    2. scores = tanh(scores/softcap) * softcap   (only on valid positions)
    3. scores[masked] = -inf                       (re-apply mask after softcap)
    4. weights = softmax(scores)

    Args:
        log_counts: (B, S) log1p(counts + eps)
        pad_mask: (B, S) True = valid token, False = padding
        causal: apply causal masking

    Returns:
        count_bias: (B, 1, S, S) additive bias (0 for masked positions, NOT -inf)
        attn_mask:  (B, 1, S, S) bool, True = allowed, False = masked
    """
    B, S = log_counts.shape
    device = log_counts.device
    dtype = log_counts.dtype

    q_pos = torch.arange(S, device=device)
    kv_pos = torch.arange(S, device=device)

    # Bias mask: when emb_mode=True, apply everywhere; otherwise only q > kv
    if emb_mode:
        bias_mask = torch.ones(S, S, device=device, dtype=torch.bool)
    else:
        bias_mask = (q_pos.unsqueeze(1) > kv_pos.unsqueeze(0))  # strictly-after

    # Broadcast log_counts along KV dim, zero where bias_mask is False
    count_bias = log_counts[:, None, None, :].expand(-1, -1, S, -1)
    count_bias = count_bias * bias_mask[None, None, :, :].to(dtype)

    # Combined boolean mask: causal AND not-padded
    if causal:
        causal_ok = q_pos.unsqueeze(1) >= kv_pos.unsqueeze(0)  # (S, S)
    else:
        causal_ok = torch.ones(S, S, device=device, dtype=torch.bool)

    pad_ok = pad_mask[:, None, None, :]  # (B, 1, 1, S)
    attn_mask = causal_ok[None, None, :, :] & pad_ok  # (B, 1, S, S)

    # Zero out bias on masked positions (will be -inf'd AFTER softcap)
    count_bias = count_bias.masked_fill(~attn_mask, 0.0)

    return count_bias, attn_mask


# Keep old name for PoC backward compatibility
def build_attn_bias(log_counts, pad_mask, causal=True, emb_mode=False):
    """Legacy: returns combined bias with -inf (only correct when softcap=0)."""
    count_bias, attn_mask = build_attn_bias_and_mask(log_counts, pad_mask, causal, emb_mode=emb_mode)
    count_bias = count_bias.masked_fill(~attn_mask, float("-inf"))
    return count_bias


def build_decode_bias(
    log_counts_so_far: Tensor,
    current_pos: int,
) -> Tensor:
    """
    Build attention bias for single-token decode step.

    Args:
        log_counts_so_far: (B, kv_len) log-counts for all positions up to current
        current_pos: the query position index (0-based)

    Returns:
        (B, 1, 1, kv_len) attention bias
    """
    B, kv_len = log_counts_so_far.shape
    device = log_counts_so_far.device
    dtype = log_counts_so_far.dtype

    bias = log_counts_so_far[:, None, None, :]  # (B, 1, 1, kv_len)

    # Zero out diagonal: position current_pos is not "strictly after" itself
    # All prior positions (0..current_pos-1) get the bias, current_pos gets 0
    mask = torch.ones(1, 1, 1, kv_len, device=device, dtype=dtype)
    mask[0, 0, 0, current_pos] = 0.0
    bias = bias * mask

    return bias


class SDPAAttention(nn.Module):
    """
    Multi-head self-attention with softcap and KV cache.
    Weight-compatible with TranscriptFormer's MultiHeadSelfFlexAttn.
    """

    def __init__(self, d_model: int, nheads: int, bias: bool = False):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.h = nheads
        # Same structure as original: linears[0]=Q, [1]=K, [2]=V, [3]=Out
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=bias) for _ in range(4)
        ])

    def forward(
        self,
        inp: Tensor,
        count_bias: Tensor = None,
        attn_mask: Tensor = None,
        softcap: float = 0.0,
        past_key_value: tuple = None,
        use_cache: bool = False,
    ):
        """
        Args:
            inp: (B, S_q, D) input
            count_bias: (B, 1, S_q, S_kv) additive log-count bias (finite values only)
            attn_mask: (B, 1, S_q, S_kv) bool, True=allowed, False=masked
            softcap: softcap value
            past_key_value: (past_k, past_v) for KV cache
            use_cache: return updated KV cache
        """
        B = inp.size(0)
        q = self.linears[0](inp).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linears[1](inp).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linears[2](inp).view(B, -1, self.h, self.d_k).transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present = (k, v) if use_cache else None

        # Attention: bias → softcap → mask → softmax
        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if count_bias is not None:
            scores = scores + count_bias

        if softcap > 0:
            scores = torch.tanh(scores / softcap) * softcap

        # Apply mask AFTER softcap to preserve -inf
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        o = torch.matmul(weights, v)

        o = o.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linears[3](o), present


class SDPATransformerLayer(nn.Module):
    """
    Drop-in replacement for FlexAttnTransformerLayer.
    Same pre-norm architecture, compatible weight names.
    """

    def __init__(self, d_model, nhead, dim_fw=2048, dropout=0.0,
                 fw_bias=False, attn_bias=False, activation="gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_fw, bias=fw_bias)
        self.linear2 = nn.Linear(dim_fw, d_model, bias=fw_bias)
        self.self_attn = SDPAAttention(d_model, nhead, bias=attn_bias)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        activations = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        self.activation = activations[activation]()

    def forward(self, x, count_bias=None, attn_mask=None, softcap=0.0,
                past_key_value=None, use_cache=False):
        # NOTE: Original TranscriptFormer uses an unusual residual pattern:
        #   x = norm(x); x = attn(x) + x   (residual adds NORMED x, not original)
        # We replicate this exactly for weight compatibility.
        x = self.norm1(x)
        attn_out, present = self.self_attn(
            x, count_bias=count_bias, attn_mask=attn_mask,
            softcap=softcap, past_key_value=past_key_value, use_cache=use_cache,
        )
        x = attn_out + x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x)))) + x
        return x, present


class SDPATranscriptEncoder(nn.Module):
    """Drop-in replacement for TranscriptEncoder with KV cache."""

    def __init__(self, embed_dim, num_head, nlayers, model_dim=2048,
                 dropout=0.0, activation="gelu", attn_bias=False, fw_bias=False):
        super().__init__()
        self.nlayers = nlayers
        self.nheads = num_head
        self.encoder_layers = nn.ModuleList([
            SDPATransformerLayer(
                d_model=embed_dim, nhead=num_head, dim_fw=model_dim,
                dropout=dropout, fw_bias=fw_bias, attn_bias=attn_bias,
                activation=activation,
            )
            for _ in range(nlayers)
        ])

    def forward(self, x, count_bias=None, attn_mask=None, softcap=0.0,
                past_key_values=None, use_cache=False):
        presents = [] if use_cache else None
        for i, layer in enumerate(self.encoder_layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            x, present = layer(
                x, count_bias=count_bias, attn_mask=attn_mask,
                softcap=softcap, past_key_value=pkv, use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)
        return x, presents
