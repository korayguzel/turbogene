"""
SDPA-converted TranscriptFormer model.

Loads original TranscriptFormer weights and replaces FlexAttention
with SDPA attention layers supporting KV cache.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from turbogene.sdpa_layers import (
    SDPATranscriptEncoder,
    build_attn_bias_and_mask,
    build_decode_bias,
)

logger = logging.getLogger(__name__)


def convert_encoder_state_dict(orig_state_dict: dict) -> dict:
    """
    Convert FlexAttention encoder state dict to SDPA encoder format.
    Weight names are identical — no renaming needed.
    """
    # The weight names are the same between FlexAttn and SDPA versions:
    # encoder_layers.{i}.self_attn.linears.{0-3}.weight
    # encoder_layers.{i}.norm1.{weight,bias}
    # encoder_layers.{i}.norm2.{weight,bias}
    # encoder_layers.{i}.linear1.{weight,bias}
    # encoder_layers.{i}.linear2.{weight,bias}
    return orig_state_dict


class SDPATranscriptformer(nn.Module):
    """
    TranscriptFormer with SDPA attention and KV cache support.

    This is a thin wrapper that:
    1. Holds the same components as the original Transcriptformer
    2. Replaces TranscriptEncoder with SDPATranscriptEncoder
    3. Implements forward() with pre-computed attn_bias instead of score_mod
    """

    def __init__(self, original_model):
        """
        Initialize from an already-loaded original Transcriptformer model.
        Copies all components and replaces the encoder.
        """
        super().__init__()

        self.model_config = original_model.model_config
        self.loss_config = original_model.loss_config
        self.gene_vocab = original_model.gene_vocab

        # Copy non-encoder components directly
        self.gene_embeddings = original_model.gene_embeddings
        self.mu = original_model.mu
        self.criterion = original_model.criterion

        if hasattr(original_model, "gene_id_head"):
            self.gene_id_head = original_model.gene_id_head
        if hasattr(original_model, "gene_id_criterion"):
            self.gene_id_criterion = original_model.gene_id_criterion
        if hasattr(original_model, "aux_embeddings"):
            self.aux_embeddings = original_model.aux_embeddings

        # Create SDPA encoder with same architecture
        mc = self.model_config
        self.transformer_encoder = SDPATranscriptEncoder(
            embed_dim=mc.embed_dim,
            num_head=mc.num_heads,
            nlayers=mc.num_layers,
            model_dim=mc.model_dim,
            dropout=mc.dropout,
            activation=mc.activation,
            attn_bias=mc.attn_bias,
            fw_bias=mc.fw_bias,
        )

        # Copy encoder weights (names are compatible)
        orig_enc_state = original_model.transformer_encoder.state_dict()
        converted = convert_encoder_state_dict(orig_enc_state)
        self.transformer_encoder.load_state_dict(converted)

        logger.info(
            f"Converted {mc.num_layers} layers from FlexAttention to SDPA "
            f"(embed_dim={mc.embed_dim}, heads={mc.num_heads})"
        )

    def _pad_mask(self, gene_tokens, aux_tokens=None, dtype="float"):
        pad_idx = self.gene_vocab.pad_idx
        pad_mask = gene_tokens == pad_idx
        if aux_tokens is not None:
            aux_vocab = self.aux_embeddings
            # Simplified: aux tokens are not padded in typical inference
            aux_pad_mask = torch.zeros(
                aux_tokens.shape[0], aux_tokens.shape[1],
                device=aux_tokens.device, dtype=torch.bool,
            )
            pad_mask = torch.cat([aux_pad_mask, pad_mask], dim=1)
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
        return (
            self.gene_embeddings(right_shifted),
            self.gene_embeddings(gene_token_indices),
        )

    def get_aux_embeddings(self, aux_token_indices):
        aux_embs = []
        for i, emb_layer in enumerate(self.aux_embeddings.values()):
            aux_embs.append(emb_layer(aux_token_indices[:, i]))
        return torch.stack(aux_embs, dim=1)

    def forward(
        self,
        gene_token_indices: Tensor,
        gene_counts: Tensor,
        aux_token_indices: Tensor = None,
        embed: bool = False,
        use_cache: bool = False,
        past_key_values: list = None,
    ) -> dict:
        """
        Forward pass with SDPA attention and optional KV cache.

        Args:
            gene_token_indices: (B, S) gene token IDs
            gene_counts: (B, S) raw expression counts
            aux_token_indices: (B, N_aux) optional auxiliary token IDs
            embed: return cell embeddings
            use_cache: return KV cache for incremental decode
            past_key_values: list of (K, V) tuples from previous steps
        """
        mc = self.model_config

        # Right-shift for autoregressive
        right_shifted_gene_tokens = torch.cat([
            self.gene_vocab.start_idx * torch.ones_like(gene_token_indices[:, :1]),
            gene_token_indices[:, :-1],
        ], dim=1)
        right_shifted_counts = torch.cat([
            torch.ones_like(gene_counts[:, :1]),
            gene_counts[:, :-1],
        ], dim=1)

        # Embeddings
        right_shifted_embs, gene_embs = self.get_gene_embeddings(gene_token_indices)

        # Aux tokens
        if aux_token_indices is not None:
            aux_embs = self.get_aux_embeddings(aux_token_indices)
            right_shifted_embs = torch.cat([aux_embs, right_shifted_embs], dim=1)
            right_shifted_counts = torch.cat([
                torch.ones_like(aux_token_indices).float(),
                right_shifted_counts,
            ], dim=1)

        # Build pad mask (bool: True=valid)
        pad_mask = self._pad_mask(right_shifted_gene_tokens, aux_token_indices, dtype="bool")

        # Log counts for attention bias
        log_counts = torch.log1p(right_shifted_counts + mc.log_counts_eps)

        # Build attention bias and mask (separate for softcap correctness)
        # emb_mode=embed: when True, log-count bias applies everywhere (not just strictly-after)
        count_bias, attn_mask = build_attn_bias_and_mask(
            log_counts, pad_mask, causal=True, emb_mode=embed,
        )

        # Encoder forward
        transformer_output, kv_cache = self.transformer_encoder(
            x=right_shifted_embs,
            count_bias=count_bias,
            attn_mask=attn_mask,
            softcap=mc.softcap,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Strip aux tokens from output
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

        # Gene ID targets
        result["input_gene_token_indices"] = torch.cat([
            gene_token_indices[:, :-1],
            self.gene_vocab.end_idx * torch.ones_like(gene_token_indices[:, :1]),
        ], dim=1)

        result["input_counts"] = torch.cat([
            gene_counts[:, :-1],
            torch.ones_like(gene_counts[:, :1]),
        ], dim=1)

        # Count prediction (mu)
        conditioned_output = torch.cat([gene_output, gene_embs], dim=-1)
        result["mu"] = self.mu(
            gene_output=conditioned_output,
            gene_tokens=gene_token_indices,
            mask=self._pad_mask(gene_token_indices, dtype="float"),
        )
        if mc.mu_link_fn == "softmax":
            result["mu"] = result["mu"] * gene_counts.sum(dim=1, keepdim=True)

        result["mask"] = ~self._pad_mask(gene_token_indices, dtype="bool")

        # Gene ID prediction
        if self.loss_config.gene_id_loss_weight > 0 and hasattr(self, "gene_id_head"):
            result["gene_logit"] = self.gene_id_head(gene_output)

        if use_cache:
            result["past_key_values"] = kv_cache

        return result
