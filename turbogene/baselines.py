"""
Baseline KV cache quantization methods for comparison.

Implements KIVI (2-bit asymmetric) as the primary baseline.
Reference: arXiv:2402.02750
"""

import torch
from torch import Tensor


class KIVIQuantizer:
    """KIVI: 2-bit asymmetric KV cache quantization.

    KIVI uses per-channel quantization for keys and per-token quantization
    for values, with asymmetric min-max scaling.

    Args:
        n_bits: quantization bits (default 2)
        group_size: group size for grouped quantization (0 = no grouping)
    """

    def __init__(self, n_bits: int = 2, group_size: int = 0):
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.group_size = group_size

    def _asymmetric_quantize(self, x: Tensor, dim: int) -> tuple:
        """Asymmetric min-max quantization along given dimension.

        Returns:
            indices: quantized indices (uint8)
            scale: per-group scale factors
            zero_point: per-group zero points
        """
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)

        scale = (x_max - x_min) / (self.n_levels - 1)
        scale = scale.clamp(min=1e-8)
        zero_point = x_min

        indices = ((x - zero_point) / scale).round().clamp(0, self.n_levels - 1).to(torch.uint8)
        return indices, scale, zero_point

    def _asymmetric_dequantize(self, indices: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return indices.float() * scale + zero_point

    def quantize_keys(self, k: Tensor) -> tuple:
        """Quantize keys with per-channel quantization.

        KIVI quantizes K per-channel: each dimension (head_dim) gets its own
        scale/zero_point across all sequence positions.

        Args:
            k: (B, H, S, D) key tensor

        Returns:
            indices, scale, zero_point
        """
        # Per-channel: quantize along sequence dim (dim=2)
        return self._asymmetric_quantize(k, dim=2)

    def quantize_values(self, v: Tensor) -> tuple:
        """Quantize values with per-token quantization.

        KIVI quantizes V per-token: each position gets its own
        scale/zero_point across all dimensions.

        Args:
            v: (B, H, S, D) value tensor

        Returns:
            indices, scale, zero_point
        """
        # Per-token: quantize along head_dim (dim=3)
        return self._asymmetric_quantize(v, dim=3)

    def dequantize_keys(self, indices: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self._asymmetric_dequantize(indices, scale, zero_point)

    def dequantize_values(self, indices: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self._asymmetric_dequantize(indices, scale, zero_point)

    def memory_bytes(self, batch_size: int, seq_len: int, num_layers: int,
                     num_heads: int, head_dim: int) -> dict:
        B, S, L, H, D = batch_size, seq_len, num_layers, num_heads, head_dim

        # Indices: 2-bit packed
        idx_bytes = 2 * L * B * H * S * D * self.n_bits / 8

        # K scales/zp: per-channel → (B, H, 1, D) per layer → S drops out
        k_meta = 2 * L * B * H * 1 * D * 2 * 2  # scale + zp, float16

        # V scales/zp: per-token → (B, H, S, 1) per layer → D drops out
        v_meta = 2 * L * B * H * S * 1 * 2 * 2

        total = idx_bytes + k_meta + v_meta
        fp16_kv = 2 * L * B * H * S * D * 2

        return {
            "total_bytes": total,
            "fp16_kv_bytes": fp16_kv,
            "compression_ratio": fp16_kv / total if total > 0 else 0,
            "effective_bits": total * 8 / (2 * L * B * H * S * D) if D > 0 else 0,
        }
