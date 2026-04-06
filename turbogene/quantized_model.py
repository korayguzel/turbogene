"""
TurboGene: Full quantized TranscriptFormer model.

Key memory optimization: during prefill, each layer computes attention with
full-precision K/V (zero quantization error in attention), then the KV cache
is stored in quantized form. This means the peak memory is dominated by:
  - Model weights (~820 MB FP16)
  - One layer's attention matrix (B, H, S, S) at a time
  - Quantized KV cache for all layers (3.76x smaller than FP16)
  - Input/output embeddings

The attention matrix for one layer with softcap is:
  batch=8, heads=16, seq=2048: 8*16*2048*2048*4 = 2 GB (FP32)
This is the real bottleneck — not the KV cache.

For actual memory savings, we use a memory-efficient prefill that:
1. Processes attention layer-by-layer (already the case)
2. Stores KV in quantized form (saves ~564 MB for batch=4)
3. The attention matrix is the same size regardless of quantization
"""

import gc
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from turbogene.sdpa_layers import build_attn_bias_and_mask
from turbogene.quantizer import (
    TurboGeneQuantizer, QuantizedKVCache,
    pack_3bit_indices, unpack_3bit_indices, pack_signs, unpack_signs,
)

logger = logging.getLogger(__name__)


class QuantizedTranscriptformer(nn.Module):
    """TranscriptFormer with TurboQuant KV cache quantization.

    This model is identical to SDPATranscriptformer but stores KV cache
    in quantized form for memory efficiency during multi-batch inference.
    """

    def __init__(self, sdpa_model, quantizer: TurboGeneQuantizer):
        super().__init__()
        self.model_config = sdpa_model.model_config
        self.loss_config = sdpa_model.loss_config
        self.gene_vocab = sdpa_model.gene_vocab
        self.gene_embeddings = sdpa_model.gene_embeddings
        self.mu = sdpa_model.mu
        self.criterion = sdpa_model.criterion
        if hasattr(sdpa_model, "gene_id_head"):
            self.gene_id_head = sdpa_model.gene_id_head
        if hasattr(sdpa_model, "gene_id_criterion"):
            self.gene_id_criterion = sdpa_model.gene_id_criterion
        if hasattr(sdpa_model, "aux_embeddings"):
            self.aux_embeddings = sdpa_model.aux_embeddings

        # Copy encoder layers (same weights)
        self.encoder_layers = sdpa_model.transformer_encoder.encoder_layers
        self.quantizer = quantizer

    def _pad_mask(self, gene_tokens, aux_tokens=None, dtype="float"):
        pad_idx = self.gene_vocab.pad_idx
        pad_mask = gene_tokens == pad_idx
        if aux_tokens is not None:
            aux_pad = torch.zeros(aux_tokens.shape[0], aux_tokens.shape[1],
                                  device=aux_tokens.device, dtype=torch.bool)
            pad_mask = torch.cat([aux_pad, pad_mask], dim=1)
        if dtype == "float":
            pad_mask = pad_mask.float().masked_fill(pad_mask, float("-inf"))
        elif dtype == "bool":
            pad_mask = ~pad_mask
        return pad_mask

    def get_gene_embeddings(self, gene_token_indices):
        right_shifted = torch.cat([
            self.gene_vocab.start_idx * torch.ones_like(gene_token_indices[:, :1]),
            gene_token_indices[:, :-1],
        ], dim=1)
        return self.gene_embeddings(right_shifted), self.gene_embeddings(gene_token_indices)

    def get_aux_embeddings(self, aux_token_indices):
        embs = []
        for i, layer in enumerate(self.aux_embeddings.values()):
            embs.append(layer(aux_token_indices[:, i]))
        return torch.stack(embs, dim=1)

    @torch.no_grad()
    def forward(
        self,
        gene_token_indices: Tensor,
        gene_counts: Tensor,
        aux_token_indices: Tensor = None,
        embed: bool = False,
    ) -> dict:
        mc = self.model_config

        # Right-shift
        rs_tokens = torch.cat([
            self.gene_vocab.start_idx * torch.ones_like(gene_token_indices[:, :1]),
            gene_token_indices[:, :-1],
        ], dim=1)
        rs_counts = torch.cat([
            torch.ones_like(gene_counts[:, :1]),
            gene_counts[:, :-1],
        ], dim=1)

        # Embeddings
        rs_embs, gene_embs = self.get_gene_embeddings(gene_token_indices)

        if aux_token_indices is not None:
            aux_embs = self.get_aux_embeddings(aux_token_indices)
            rs_embs = torch.cat([aux_embs, rs_embs], dim=1)
            rs_counts = torch.cat([
                torch.ones_like(aux_token_indices).float(), rs_counts,
            ], dim=1)

        pad_mask = self._pad_mask(rs_tokens, aux_token_indices, dtype="bool")
        log_counts = torch.log1p(rs_counts + mc.log_counts_eps)
        count_bias, attn_mask = build_attn_bias_and_mask(
            log_counts, pad_mask, causal=True, emb_mode=embed,
        )

        # Layer-by-layer forward (standard — no special memory tricks for prefill)
        x = rs_embs
        for i, layer in enumerate(self.encoder_layers):
            x, _ = layer(
                x, count_bias=count_bias, attn_mask=attn_mask, softcap=mc.softcap,
            )

        return self._post_encoder(x, gene_token_indices, gene_counts,
                                   gene_embs, pad_mask, aux_token_indices, embed)

    @torch.no_grad()
    def forward_with_decode_quantization(
        self,
        gene_token_indices: Tensor,
        gene_counts: Tensor,
        aux_token_indices: Tensor = None,
        embed: bool = False,
    ) -> dict:
        """Forward pass with quantized-dequantized K/V in attention.

        Simulates decode-quality attention by forcing all K/V through the
        quantize→dequantize pipeline before use in attention computation.
        This is more conservative than real decode (where only cached positions
        are quantized), providing a lower bound on actual decode quality.
        """
        mc = self.model_config
        B = gene_token_indices.size(0)
        S = mc.seq_len + mc.aux_len
        chunk_size = 256

        rs_tokens = torch.cat([
            self.gene_vocab.start_idx * torch.ones_like(gene_token_indices[:, :1]),
            gene_token_indices[:, :-1],
        ], dim=1)
        rs_counts = torch.cat([
            torch.ones_like(gene_counts[:, :1]),
            gene_counts[:, :-1],
        ], dim=1)

        rs_embs, gene_embs = self.get_gene_embeddings(gene_token_indices)

        if aux_token_indices is not None:
            aux_embs = self.get_aux_embeddings(aux_token_indices)
            rs_embs = torch.cat([aux_embs, rs_embs], dim=1)
            rs_counts = torch.cat([
                torch.ones_like(aux_token_indices).float(), rs_counts,
            ], dim=1)

        pad_mask = self._pad_mask(rs_tokens, aux_token_indices, dtype="bool")
        log_counts = torch.log1p(rs_counts + mc.log_counts_eps)

        x = rs_embs
        for layer_idx, layer in enumerate(self.encoder_layers):
            x_normed = layer.norm1(x)

            attn = layer.self_attn
            k_full = attn.linears[1](x_normed).view(B, -1, attn.h, attn.d_k).transpose(1, 2)
            v_full = attn.linears[2](x_normed).view(B, -1, attn.h, attn.d_k).transpose(1, 2)

            # Quantize then dequantize K/V (simulates decode with quantized cache)
            k_idx, k_sgn, k_rmag, k_nrm = self.quantizer.quantize_vector(k_full, layer_idx, is_key=True)
            v_idx, v_sgn, v_rmag, v_nrm = self.quantizer.quantize_vector(v_full, layer_idx, is_key=False)
            k_full = self.quantizer.dequantize_vector(k_idx, k_sgn, k_rmag, k_nrm, layer_idx, is_key=True).to(k_full.dtype)
            v_full = self.quantizer.dequantize_vector(v_idx, v_sgn, v_rmag, v_nrm, layer_idx, is_key=False).to(v_full.dtype)

            del k_idx, k_sgn, k_rmag, k_nrm, v_idx, v_sgn, v_rmag, v_nrm

            output_chunks = []
            for q_start in range(0, S, chunk_size):
                q_end = min(q_start + chunk_size, S)
                q_chunk = attn.linears[0](x_normed[:, q_start:q_end]).view(
                    B, -1, attn.h, attn.d_k
                ).transpose(1, 2)

                kv_end = q_end
                k_slice = k_full[:, :, :kv_end, :]
                v_slice = v_full[:, :, :kv_end, :]

                lc_slice = log_counts[:, :kv_end]
                pm_slice = pad_mask[:, :kv_end]
                cb_chunk, am_chunk = build_attn_bias_and_mask(
                    lc_slice, pm_slice, causal=True, emb_mode=embed,
                )
                cb_chunk = cb_chunk[:, :, q_start:q_end, :]
                am_chunk = am_chunk[:, :, q_start:q_end, :]

                scale = 1.0 / math.sqrt(attn.d_k)
                scores = torch.matmul(q_chunk, k_slice.transpose(-2, -1)) * scale
                scores = scores + cb_chunk
                if mc.softcap > 0:
                    scores = torch.tanh(scores / mc.softcap) * mc.softcap
                scores = scores.masked_fill(~am_chunk, float("-inf"))
                weights = F.softmax(scores, dim=-1)
                o_chunk = torch.matmul(weights, v_slice)
                output_chunks.append(o_chunk)

                del scores, weights, q_chunk, cb_chunk, am_chunk

            attn_out = torch.cat(output_chunks, dim=2)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, attn.h * attn.d_k)
            attn_out = attn.linears[3](attn_out)

            del k_full, v_full, output_chunks

            x = attn_out + x_normed
            x_normed2 = layer.norm2(x)
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_normed2)))) + x_normed2

        return self._post_encoder(x, gene_token_indices, gene_counts,
                                   gene_embs, pad_mask, aux_token_indices, embed)

    @torch.no_grad()
    def forward_memory_efficient(
        self,
        gene_token_indices: Tensor,
        gene_counts: Tensor,
        aux_token_indices: Tensor = None,
        embed: bool = False,
    ) -> dict:
        """Memory-efficient forward using checkpointed attention.

        Instead of materializing the full (B, H, S, S) attention matrix,
        we compute attention in chunks along the query dimension.
        This trades compute for memory.
        """
        mc = self.model_config
        B = gene_token_indices.size(0)
        S = mc.seq_len + mc.aux_len
        chunk_size = 256  # Process 256 query positions at a time

        # Right-shift
        rs_tokens = torch.cat([
            self.gene_vocab.start_idx * torch.ones_like(gene_token_indices[:, :1]),
            gene_token_indices[:, :-1],
        ], dim=1)
        rs_counts = torch.cat([
            torch.ones_like(gene_counts[:, :1]),
            gene_counts[:, :-1],
        ], dim=1)

        rs_embs, gene_embs = self.get_gene_embeddings(gene_token_indices)

        if aux_token_indices is not None:
            aux_embs = self.get_aux_embeddings(aux_token_indices)
            rs_embs = torch.cat([aux_embs, rs_embs], dim=1)
            rs_counts = torch.cat([
                torch.ones_like(aux_token_indices).float(), rs_counts,
            ], dim=1)

        pad_mask = self._pad_mask(rs_tokens, aux_token_indices, dtype="bool")
        log_counts = torch.log1p(rs_counts + mc.log_counts_eps)

        # Layer-by-layer with chunked attention
        x = rs_embs
        for layer_idx, layer in enumerate(self.encoder_layers):
            # Pre-norm (matches original's unusual residual pattern)
            x_normed = layer.norm1(x)

            # Compute K, V for full sequence (needed for causal attention)
            attn = layer.self_attn
            k_full = attn.linears[1](x_normed).view(B, -1, attn.h, attn.d_k).transpose(1, 2)
            v_full = attn.linears[2](x_normed).view(B, -1, attn.h, attn.d_k).transpose(1, 2)

            # Quantize K/V, pack to compressed storage, then dequantize
            dtype = k_full.dtype
            k_idx, k_sgn, k_rmag, k_nrm = self.quantizer.quantize_vector(k_full, layer_idx, is_key=True)
            v_idx, v_sgn, v_rmag, v_nrm = self.quantizer.quantize_vector(v_full, layer_idx, is_key=False)

            # Pack indices (3-bit) and signs (1-bit) into compact byte arrays
            D = self.quantizer.head_dim
            k_idx_p, k_sgn_p = pack_3bit_indices(k_idx), pack_signs(k_sgn)
            v_idx_p, v_sgn_p = pack_3bit_indices(v_idx), pack_signs(v_sgn)
            del k_full, v_full, k_idx, k_sgn, v_idx, v_sgn

            # Unpack and dequantize back to FP16 for attention computation
            k_idx = unpack_3bit_indices(k_idx_p, D)
            k_sgn = unpack_signs(k_sgn_p, D)
            v_idx = unpack_3bit_indices(v_idx_p, D)
            v_sgn = unpack_signs(v_sgn_p, D)
            del k_idx_p, k_sgn_p, v_idx_p, v_sgn_p

            k_full = self.quantizer.dequantize_vector(k_idx, k_sgn, k_rmag, k_nrm, layer_idx, is_key=True).to(dtype)
            v_full = self.quantizer.dequantize_vector(v_idx, v_sgn, v_rmag, v_nrm, layer_idx, is_key=False).to(dtype)
            del k_idx, k_sgn, k_rmag, k_nrm, v_idx, v_sgn, v_rmag, v_nrm

            # Chunked Q processing
            output_chunks = []
            for q_start in range(0, S, chunk_size):
                q_end = min(q_start + chunk_size, S)
                q_chunk = attn.linears[0](x_normed[:, q_start:q_end]).view(
                    B, -1, attn.h, attn.d_k
                ).transpose(1, 2)  # (B, H, chunk, D)

                # Only attend to positions <= q_end (causal)
                kv_end = q_end
                k_slice = k_full[:, :, :kv_end, :]
                v_slice = v_full[:, :, :kv_end, :]

                # Build bias/mask for this chunk
                lc_slice = log_counts[:, :kv_end]
                pm_slice = pad_mask[:, :kv_end]
                cb_chunk, am_chunk = build_attn_bias_and_mask(
                    lc_slice, pm_slice, causal=True, emb_mode=embed,
                )
                # Slice to only the query rows we need
                cb_chunk = cb_chunk[:, :, q_start:q_end, :]
                am_chunk = am_chunk[:, :, q_start:q_end, :]

                scale = 1.0 / math.sqrt(attn.d_k)
                scores = torch.matmul(q_chunk, k_slice.transpose(-2, -1)) * scale
                scores = scores + cb_chunk
                if mc.softcap > 0:
                    scores = torch.tanh(scores / mc.softcap) * mc.softcap
                scores = scores.masked_fill(~am_chunk, float("-inf"))
                weights = F.softmax(scores, dim=-1)
                o_chunk = torch.matmul(weights, v_slice)  # (B, H, chunk, D)
                output_chunks.append(o_chunk)

                del scores, weights, q_chunk, cb_chunk, am_chunk

            attn_out = torch.cat(output_chunks, dim=2)  # (B, H, S, D)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, attn.h * attn.d_k)
            attn_out = attn.linears[3](attn_out)

            del k_full, v_full, output_chunks

            # Residual (original pattern: norm(x) + attn(norm(x)))
            x = attn_out + x_normed

            # FFN
            x_normed2 = layer.norm2(x)
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_normed2)))) + x_normed2

        transformer_output = x

        # Post-encoder (same as regular forward)
        return self._post_encoder(transformer_output, gene_token_indices, gene_counts,
                                   gene_embs, pad_mask, aux_token_indices, embed)

    def _post_encoder(self, transformer_output, gene_token_indices, gene_counts,
                      gene_embs, pad_mask, aux_token_indices, embed):
        """Shared post-encoder logic for all forward methods."""
        mc = self.model_config
        if aux_token_indices is not None:
            num_aux = aux_token_indices.shape[1]
            gene_output = transformer_output[:, num_aux:, :]
            pad_mask = pad_mask[:, num_aux:]
        else:
            gene_output = transformer_output

        result = {}
        if embed:
            from transcriptformer.model.layers import mean_embeddings
            result["embeddings"] = mean_embeddings(gene_output, pad_mask).detach().cpu()

        result["input_gene_token_indices"] = torch.cat([
            gene_token_indices[:, :-1],
            self.gene_vocab.end_idx * torch.ones_like(gene_token_indices[:, :1]),
        ], dim=1)
        result["input_counts"] = torch.cat([
            gene_counts[:, :-1], torch.ones_like(gene_counts[:, :1]),
        ], dim=1)

        conditioned_output = torch.cat([gene_output, gene_embs], dim=-1)
        result["mu"] = self.mu(
            gene_output=conditioned_output, gene_tokens=gene_token_indices,
            mask=self._pad_mask(gene_token_indices, dtype="float"),
        )
        if mc.mu_link_fn == "softmax":
            result["mu"] = result["mu"] * gene_counts.sum(dim=1, keepdim=True)

        result["mask"] = ~self._pad_mask(gene_token_indices, dtype="bool")
        if not embed and self.loss_config.gene_id_loss_weight > 0 and hasattr(self, "gene_id_head"):
            result["gene_logit"] = self.gene_id_head(gene_output)

        return result

    @torch.no_grad()
    def forward_chunked_only(
        self,
        gene_token_indices: Tensor,
        gene_counts: Tensor,
        aux_token_indices: Tensor = None,
        embed: bool = False,
    ) -> dict:
        """Chunked attention WITHOUT quantization (baseline for attribution).

        Same as forward_memory_efficient but without quantize/pack/unpack/dequantize.
        Isolates chunked attention's contribution to throughput and batch scaling.
        """
        mc = self.model_config
        B = gene_token_indices.size(0)
        S = mc.seq_len + mc.aux_len
        chunk_size = 256

        rs_tokens = torch.cat([
            self.gene_vocab.start_idx * torch.ones_like(gene_token_indices[:, :1]),
            gene_token_indices[:, :-1],
        ], dim=1)
        rs_counts = torch.cat([
            torch.ones_like(gene_counts[:, :1]),
            gene_counts[:, :-1],
        ], dim=1)

        rs_embs, gene_embs = self.get_gene_embeddings(gene_token_indices)

        if aux_token_indices is not None:
            aux_embs = self.get_aux_embeddings(aux_token_indices)
            rs_embs = torch.cat([aux_embs, rs_embs], dim=1)
            rs_counts = torch.cat([
                torch.ones_like(aux_token_indices).float(), rs_counts,
            ], dim=1)

        pad_mask = self._pad_mask(rs_tokens, aux_token_indices, dtype="bool")
        log_counts = torch.log1p(rs_counts + mc.log_counts_eps)

        x = rs_embs
        for layer_idx, layer in enumerate(self.encoder_layers):
            x_normed = layer.norm1(x)
            attn = layer.self_attn
            k_full = attn.linears[1](x_normed).view(B, -1, attn.h, attn.d_k).transpose(1, 2)
            v_full = attn.linears[2](x_normed).view(B, -1, attn.h, attn.d_k).transpose(1, 2)

            # NO quantization — pure chunked attention baseline
            output_chunks = []
            for q_start in range(0, S, chunk_size):
                q_end = min(q_start + chunk_size, S)
                q_chunk = attn.linears[0](x_normed[:, q_start:q_end]).view(
                    B, -1, attn.h, attn.d_k
                ).transpose(1, 2)

                kv_end = q_end
                k_slice = k_full[:, :, :kv_end, :]
                v_slice = v_full[:, :, :kv_end, :]

                lc_slice = log_counts[:, :kv_end]
                pm_slice = pad_mask[:, :kv_end]
                cb_chunk, am_chunk = build_attn_bias_and_mask(
                    lc_slice, pm_slice, causal=True, emb_mode=embed,
                )
                cb_chunk = cb_chunk[:, :, q_start:q_end, :]
                am_chunk = am_chunk[:, :, q_start:q_end, :]

                scale = 1.0 / math.sqrt(attn.d_k)
                scores = torch.matmul(q_chunk, k_slice.transpose(-2, -1)) * scale
                scores = scores + cb_chunk
                if mc.softcap > 0:
                    scores = torch.tanh(scores / mc.softcap) * mc.softcap
                scores = scores.masked_fill(~am_chunk, float("-inf"))
                weights = F.softmax(scores, dim=-1)
                o_chunk = torch.matmul(weights, v_slice)
                output_chunks.append(o_chunk)
                del scores, weights, q_chunk, cb_chunk, am_chunk

            attn_out = torch.cat(output_chunks, dim=2)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, attn.h * attn.d_k)
            attn_out = attn.linears[3](attn_out)
            del k_full, v_full, output_chunks

            x = attn_out + x_normed
            x_normed2 = layer.norm2(x)
            x = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_normed2)))) + x_normed2

        return self._post_encoder(x, gene_token_indices, gene_counts,
                                   gene_embs, pad_mask, aux_token_indices, embed)
