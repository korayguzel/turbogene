"""
TurboGene KV Cache Quantizer

Implements TurboQuant-style KV cache quantization for TranscriptFormer:
  Stage 1: Norm extraction + Orthogonal rotation (Pi) + Lloyd-Max 3-bit scalar quantization
  Stage 2: QJL 1-bit residual correction (sign(S @ residual), S ~ N(0,1))

References:
  - TurboQuant: arXiv:2504.19874
  - TurboESM:   arXiv:2603.26110
  - turboquant_plus: https://github.com/TheTom/turboquant_plus (Apache 2.0)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# QJL dequantization scaling constant (unbiased estimator)
QJL_CONST = math.sqrt(math.pi / 2)


# ─────────────────────────────────────────────────────────────
# Bit Packing Utilities
# ─────────────────────────────────────────────────────────────

def pack_signs(signs: Tensor) -> Tensor:
    """Pack bool sign tensor into uint8 bitfield.

    8 signs per byte. True -> 1, False -> 0.

    Args:
        signs: bool tensor of shape (..., D)

    Returns:
        uint8 tensor of shape (..., ceil(D/8))
    """
    shape = signs.shape
    D = shape[-1]
    flat = signs.reshape(-1, D).to(torch.uint8)
    # Pad to multiple of 8
    pad_d = (8 - D % 8) % 8
    if pad_d > 0:
        flat = F.pad(flat, (0, pad_d))
    # Pack: each group of 8 bits -> 1 byte
    N, D_padded = flat.shape
    flat = flat.reshape(N, D_padded // 8, 8)
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                          device=flat.device, dtype=torch.uint8)
    packed = (flat * powers).sum(dim=-1).to(torch.uint8)
    return packed.reshape(*shape[:-1], -1)


def unpack_signs(packed: Tensor, D: int) -> Tensor:
    """Unpack uint8 bitfield back to bool sign tensor.

    Args:
        packed: uint8 tensor from pack_signs
        D: original last dimension size

    Returns:
        bool tensor of shape (..., D)
    """
    shape = packed.shape
    flat = packed.reshape(-1, shape[-1])
    N = flat.shape[0]
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                          device=flat.device, dtype=torch.uint8)
    unpacked = ((flat.unsqueeze(-1) & powers) > 0)  # (N, packed_D, 8)
    unpacked = unpacked.reshape(N, -1)[:, :D]
    return unpacked.reshape(*shape[:-1], D)


def pack_3bit_indices(indices: Tensor) -> Tensor:
    """Pack 3-bit indices (0-7) into compact byte array.

    Every 8 indices (24 bits) pack into 3 bytes.

    Args:
        indices: uint8 tensor of shape (..., D) with values 0-7

    Returns:
        uint8 tensor of shape (..., ceil(D*3/8))
    """
    shape = indices.shape
    D = shape[-1]
    flat = indices.reshape(-1, D).to(torch.uint8)
    N = flat.shape[0]

    # Convert each index to 3 binary digits
    bits = torch.zeros(N, D * 3, device=flat.device, dtype=torch.uint8)
    bits[:, 0::3] = (flat >> 2) & 1
    bits[:, 1::3] = (flat >> 1) & 1
    bits[:, 2::3] = flat & 1

    # Pad to multiple of 8
    total_bits = D * 3
    pad_bits = (8 - total_bits % 8) % 8
    if pad_bits > 0:
        bits = F.pad(bits, (0, pad_bits))

    # Pack 8 bits per byte
    bits = bits.reshape(N, -1, 8)
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                          device=bits.device, dtype=torch.uint8)
    packed = (bits * powers).sum(dim=-1).to(torch.uint8)
    return packed.reshape(*shape[:-1], -1)


def unpack_3bit_indices(packed: Tensor, D: int) -> Tensor:
    """Unpack 3-bit indices from compact byte array.

    Args:
        packed: uint8 tensor from pack_3bit_indices
        D: original last dimension size

    Returns:
        uint8 tensor of shape (..., D) with values 0-7
    """
    shape = packed.shape
    flat = packed.reshape(-1, shape[-1])
    N = flat.shape[0]

    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                          device=flat.device, dtype=torch.uint8)
    bits = ((flat.unsqueeze(-1) & powers) > 0).to(torch.uint8)
    bits = bits.reshape(N, -1)[:, :D * 3]

    # Reconstruct 3-bit values
    indices = (bits[:, 0::3] << 2) | (bits[:, 1::3] << 1) | bits[:, 2::3]
    return indices.reshape(*shape[:-1], D)


# ─────────────────────────────────────────────────────────────
# 1. Orthogonal Rotation
# ─────────────────────────────────────────────────────────────

def generate_random_rotation(dim: int, seed: int = 0) -> Tensor:
    """Generate a random orthogonal matrix via QR decomposition.

    This is the data-oblivious rotation from TurboQuant.
    For calibrated (data-driven) rotation, use SVD on activation data.

    Args:
        dim: dimension (head_dim, e.g. 128)
        seed: random seed for reproducibility

    Returns:
        Pi: (dim, dim) orthogonal matrix where Pi^T @ Pi = I
    """
    gen = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(dim, dim, generator=gen)
    q, r = torch.linalg.qr(random_matrix)
    # Ensure proper rotation (det=+1) by fixing sign of diagonal
    d = torch.diag(r)
    q = q * torch.sign(d).unsqueeze(0)
    return q


def calibrate_svd_rotation(activations: Tensor) -> Tensor:
    """Compute data-driven rotation matrix via SVD on calibration data.

    Like TurboESM: Pi = V^T from SVD of activation matrix.
    This spreads outliers more effectively than random rotation.

    Args:
        activations: (N, dim) collected activations for one head

    Returns:
        Pi: (dim, dim) orthogonal matrix
    """
    # Center the data
    mean = activations.mean(dim=0, keepdim=True)
    centered = activations - mean
    # SVD: X = U @ S @ V^T
    _, _, Vt = torch.linalg.svd(centered, full_matrices=True)
    return Vt  # (dim, dim)


# ─────────────────────────────────────────────────────────────
# 2. Lloyd-Max Scalar Quantization
# ─────────────────────────────────────────────────────────────

def lloyd_max_quantize(data: Tensor, n_bits: int = 3, max_iter: int = 50) -> Tensor:
    """Compute optimal Lloyd-Max centroids for scalar quantization.

    Solves the continuous 1-D k-means problem on the data distribution.

    Args:
        data: (N,) flat tensor of values to quantize
        n_bits: number of bits (2^n_bits centroids)
        max_iter: maximum Lloyd iterations

    Returns:
        centroids: (2^n_bits,) sorted centroid values
    """
    n_centroids = 2 ** n_bits
    data = data.float()

    # Initialize centroids from quantiles
    quantiles = torch.linspace(0, 1, n_centroids + 2, device=data.device)[1:-1]
    centroids = torch.quantile(data, quantiles)

    for _ in range(max_iter):
        # Assignment: find nearest centroid for each point
        dists = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # (N, C)
        assignments = dists.argmin(dim=1)  # (N,)

        # Update: new centroid = mean of assigned points
        new_centroids = torch.zeros_like(centroids)
        for c in range(n_centroids):
            mask = assignments == c
            if mask.sum() > 0:
                new_centroids[c] = data[mask].mean()
            else:
                new_centroids[c] = centroids[c]

        # Check convergence
        if torch.allclose(centroids, new_centroids, atol=1e-7):
            break
        centroids = new_centroids

    return centroids.sort().values


@dataclass
class QuantizedKVCache:
    """Stores quantized K/V cache for one layer.

    Stage 0 (Norm extraction): L2 norms stored separately (PolarQuant)
    Stage 1 (Lloyd-Max): 3-bit indices + centroids on normalized vectors
    Stage 2 (QJL): 1-bit sign(S @ residual) + residual L2 norms
    Total effective: ~3.125 bits per coordinate + norms
    """
    # Stage 1: Lloyd-Max quantized (on normalized, rotated vectors)
    k_indices: Tensor     # (B, H, S, D) uint8 -- 3-bit packed later
    v_indices: Tensor     # (B, H, S, D) uint8
    k_centroids: Tensor   # (H, 2^b) per-head centroids for K
    v_centroids: Tensor   # (H, 2^b) per-head centroids for V

    # Stage 2: QJL residual correction
    k_signs: Tensor       # (B, H, S, D) bool -- sign(S @ residual)
    v_signs: Tensor       # (B, H, S, D) bool
    k_magnitudes: Tensor  # (B, H, S) -- ||residual||_2 per position
    v_magnitudes: Tensor  # (B, H, S)

    # Norm extraction (PolarQuant)
    k_norms: Tensor       # (B, H, S) -- L2 norm of original K vectors
    v_norms: Tensor       # (B, H, S) -- L2 norm of original V vectors


class TurboGeneQuantizer(nn.Module):
    """KV cache quantizer implementing TurboQuant for TranscriptFormer.

    Quantization pipeline (per head):
      0. Extract norm: n = ||k||, k_unit = k / n     (PolarQuant)
      1. Rotate: k' = Pi @ k_unit                    (spreads outliers)
      2. Quantize: idx = nearest_centroid(k')         (3-bit Lloyd-Max)
      3. Dequantize: k_hat = centroids[idx]
      4. Residual: r = k' - k_hat
      5. QJL: signs = sign(S @ r), r_norm = ||r||     (1-bit JL projection)
      6. Reconstruct:
         correction = sqrt(pi/2)/d * r_norm * S^T @ signs
         k_rot_approx = k_hat + correction
         k_rot_approx /= ||k_rot_approx||            (norm correction)
         k_approx = Pi^T @ k_rot_approx * n           (inverse rotation + rescale)

    Args:
        num_layers: number of transformer layers
        num_heads: number of attention heads per layer
        head_dim: dimension per head
        n_bits: quantization bits (default 3 for ~3.125 effective)
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, n_bits: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits

        # Rotation matrices: Pi[layer, head] is (head_dim, head_dim)
        # Initialize with random rotations; replaced by SVD after calibration
        rotations = torch.zeros(num_layers, num_heads, head_dim, head_dim)
        for l in range(num_layers):
            for h in range(num_heads):
                rotations[l, h] = generate_random_rotation(head_dim, seed=l * num_heads + h)
        self.register_buffer("rotations", rotations)

        # QJL random projection matrix: S ~ N(0,1), shape (head_dim, head_dim)
        # Single S shared across all layers/heads (JL guarantee is per-matrix)
        gen = torch.Generator().manual_seed(12345)
        jl_matrix = torch.randn(head_dim, head_dim, generator=gen)
        self.register_buffer("jl_matrix", jl_matrix)

        # Lloyd-Max centroids: initialized to uniform, calibrated from data
        # Separate LUTs for K and V (dual LUT -- V has different outlier profile)
        k_centroids = torch.zeros(num_layers, num_heads, self.n_centroids)
        v_centroids = torch.zeros(num_layers, num_heads, self.n_centroids)

        # Default initialization: uniform in [-3, 3]
        for lh in range(num_layers * num_heads):
            uniform = torch.linspace(-3, 3, self.n_centroids)
            k_centroids.view(-1, self.n_centroids)[lh] = uniform
            v_centroids.view(-1, self.n_centroids)[lh] = uniform

        self.register_buffer("k_centroids", k_centroids)
        self.register_buffer("v_centroids", v_centroids)

        self._calibrated = False

    @torch.no_grad()
    def calibrate(self, k_activations: list, v_activations: list):
        """Calibrate rotation matrices and LUTs from real activations.

        Normalizes vectors before calibration (PolarQuant approach):
        centroids are calibrated for unit-norm vectors.

        Args:
            k_activations: list of (layer_idx, tensor) where tensor is (B, H, S, D)
            v_activations: same for values
        """
        for kv_type, activations, centroids_buf in [
            ("K", k_activations, "k_centroids"),
            ("V", v_activations, "v_centroids"),
        ]:
            for layer_idx, tensor in activations:
                B, H, S, D = tensor.shape
                for h in range(H):
                    head_data = tensor[:, h, :, :].reshape(-1, D).float()

                    # Normalize vectors before calibration (PolarQuant)
                    norms = torch.norm(head_data, dim=-1, keepdim=True)
                    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
                    head_data_normalized = head_data / safe_norms

                    # SVD-based rotation calibration (TurboESM approach)
                    if kv_type == "K":
                        Pi = calibrate_svd_rotation(head_data_normalized)
                        self.rotations[layer_idx, h] = Pi

                    # Rotate normalized data for LUT calibration
                    Pi = self.rotations[layer_idx, h]
                    rotated = head_data_normalized @ Pi.T  # (N, D)

                    # Lloyd-Max centroids on rotated normalized data
                    flat = rotated.reshape(-1)
                    centroids = lloyd_max_quantize(flat, self.n_bits)
                    getattr(self, centroids_buf)[layer_idx, h] = centroids

        self._calibrated = True

    def quantize_vector(
        self,
        x: Tensor,
        layer_idx: int,
        is_key: bool,
    ) -> tuple:
        """Quantize a K or V tensor for one layer.

        Pipeline: norm extraction -> rotation -> Lloyd-Max -> QJL residual

        Args:
            x: (B, H, S, D) key or value tensor
            layer_idx: which layer
            is_key: True for keys, False for values

        Returns:
            indices: (B, H, S, D) uint8 centroid indices
            signs: (B, H, S, D) bool -- sign(S @ residual)
            residual_norms: (B, H, S) -- ||residual||_2
            vector_norms: (B, H, S) -- ||x||_2 (original vector norms)
        """
        B, H, S, D = x.shape
        Pi = self.rotations[layer_idx]  # (H, D, D)
        centroids = self.k_centroids[layer_idx] if is_key else self.v_centroids[layer_idx]  # (H, C)

        # 0. Norm extraction (PolarQuant): separate norm from direction
        x_float = x.float()
        vector_norms = torch.norm(x_float, dim=-1)  # (B, H, S)
        safe_norms = torch.where(vector_norms > 0, vector_norms, torch.ones_like(vector_norms))
        x_normalized = x_float / safe_norms.unsqueeze(-1)  # (B, H, S, D) unit vectors

        # 1. Rotate: x_rot = x_normalized @ Pi^T
        x_rot = torch.einsum("bhsd,hde->bhse", x_normalized, Pi.transpose(-1, -2))

        # 2. Lloyd-Max quantize: find nearest centroid per coordinate
        C = self.n_centroids
        c = centroids[None, :, None, None, :]  # (1, H, 1, 1, C)
        dists = (x_rot.unsqueeze(-1) - c).abs()  # (B, H, S, D, C)
        indices = dists.argmin(dim=-1).to(torch.uint8)  # (B, H, S, D)

        # 3. Dequantize via index lookup
        idx_long = indices.long()
        c_expanded = centroids[None, :, None, None, :].expand(B, H, S, D, C)
        x_deq = torch.gather(c_expanded, dim=-1, index=idx_long.unsqueeze(-1)).squeeze(-1)

        # 4. Norm-correct x_deq to unit norm before computing residual
        # (x_rot is unit-norm; x_deq generally isn't, so normalize it first
        #  so the residual captures direction error, not norm mismatch)
        x_deq_norms = torch.norm(x_deq, dim=-1, keepdim=True)
        safe_deq_norms = torch.where(x_deq_norms > 1e-10, x_deq_norms,
                                     torch.ones_like(x_deq_norms))
        x_deq_unit = x_deq / safe_deq_norms

        # 5. QJL: 1-bit residual via random JL projection
        residual = x_rot - x_deq_unit  # (B, H, S, D) -- small, between unit vectors
        residual_norms = torch.norm(residual, dim=-1)  # (B, H, S) -- ||r||_2

        # Project through random Gaussian matrix then take sign
        # S @ r for each (b,h,s) vector: in batch form, r @ S^T
        projected = residual @ self.jl_matrix.T  # (B, H, S, D)
        signs = projected > 0  # (B, H, S, D) bool

        return indices, signs, residual_norms, vector_norms

    def dequantize_vector(
        self,
        indices: Tensor,
        signs: Tensor,
        residual_norms: Tensor,
        vector_norms: Tensor,
        layer_idx: int,
        is_key: bool,
    ) -> Tensor:
        """Dequantize and reconstruct a K or V tensor.

        Pipeline: Lloyd-Max lookup -> QJL correction -> norm correction ->
                  inverse rotation -> rescale by original norms

        Args:
            indices: (B, H, S, D) uint8
            signs: (B, H, S, D) bool -- sign(S @ residual)
            residual_norms: (B, H, S) -- ||residual||_2
            vector_norms: (B, H, S) -- original ||x||_2
            layer_idx: which layer
            is_key: True for keys

        Returns:
            x_reconstructed: (B, H, S, D) float
        """
        B, H, S, D = indices.shape
        Pi = self.rotations[layer_idx]  # (H, D, D)
        centroids = self.k_centroids[layer_idx] if is_key else self.v_centroids[layer_idx]

        # Stage 1: Lloyd-Max dequantize
        C = self.n_centroids
        c_expanded = centroids[None, :, None, None, :].expand(B, H, S, D, C)
        x_deq = torch.gather(c_expanded, dim=-1, index=indices.long().unsqueeze(-1)).squeeze(-1)

        # Norm-correct x_deq to unit norm FIRST (matches quantize_vector)
        x_deq_norms = torch.norm(x_deq, dim=-1, keepdim=True)
        safe_deq_norms = torch.where(x_deq_norms > 1e-10, x_deq_norms,
                                     torch.ones_like(x_deq_norms))
        x_deq_unit = x_deq / safe_deq_norms

        # Stage 2: QJL correction (added to norm-corrected centroid approximation)
        # Reconstruct residual: r_hat = sqrt(pi/2) / d * ||r|| * S^T @ signs
        # S^T @ signs: in batch form, signs @ S (see derivation in module docstring)
        sign_float = signs.float() * 2 - 1  # {0,1} -> {-1, +1}
        correction = sign_float @ self.jl_matrix  # (B, H, S, D) -- S^T @ signs
        scale = QJL_CONST / D * residual_norms  # (B, H, S)
        correction = correction * scale.unsqueeze(-1)  # (B, H, S, D)
        x_rot_approx = x_deq_unit + correction
        # No further norm correction: x_deq_unit is already unit-norm,
        # and the small QJL correction should improve direction, not break it

        # Inverse rotation: x = x_rot @ Pi
        x_recon = torch.einsum("bhsd,hde->bhse", x_rot_approx, Pi)

        # Rescale by original vector norms
        x_recon = x_recon * vector_norms.unsqueeze(-1)

        return x_recon

    def memory_bytes(self, batch_size: int, seq_len: int) -> dict:
        """Compute memory footprint of quantized KV cache.

        Returns dict with bytes for each component.
        """
        B, S, H, D = batch_size, seq_len, self.num_heads, self.head_dim
        L = self.num_layers

        # Stage 1: 3-bit indices packed (8 values -> 3 bytes)
        indices_bytes = 2 * L * B * H * S * D * self.n_bits / 8

        # Stage 2: 1-bit signs packed (8 signs -> 1 byte)
        signs_bytes = 2 * L * B * H * S * D / 8

        # Residual norms: 1 float16 per position per head
        residual_norm_bytes = 2 * L * B * H * S * 2  # float16

        # Vector norms: 1 float16 per position per head
        vector_norm_bytes = 2 * L * B * H * S * 2  # float16

        # Centroids: tiny, amortized (2 * L * H * 8 * 2 bytes)
        centroid_bytes = 2 * L * H * (2 ** self.n_bits) * 2

        # Rotation matrices: amortized (L * H * D * D * 2)
        rotation_bytes = L * H * D * D * 2

        # JL matrix: single D*D*4 (float32, shared)
        jl_bytes = D * D * 4

        total = (indices_bytes + signs_bytes + residual_norm_bytes +
                 vector_norm_bytes + centroid_bytes + rotation_bytes + jl_bytes)
        fp16_kv = 2 * L * B * H * S * D * 2

        return {
            "indices_bytes": indices_bytes,
            "signs_bytes": signs_bytes,
            "residual_norm_bytes": residual_norm_bytes,
            "vector_norm_bytes": vector_norm_bytes,
            "centroids_bytes": centroid_bytes,
            "rotation_bytes": rotation_bytes,
            "jl_matrix_bytes": jl_bytes,
            "total_bytes": total,
            "fp16_kv_bytes": fp16_kv,
            "compression_ratio": fp16_kv / total if total > 0 else 0,
            "effective_bits_per_coord": total * 8 / (2 * L * B * H * S * D) if D > 0 else 0,
        }


class QuantizedSDPAAttention(nn.Module):
    """SDPA attention with TurboQuant KV cache quantization.

    During prefill: compute full attention (no quantization error), store quantized KV.
    During decode: use quantized KV cache for attention.
    """

    def __init__(self, d_model: int, nheads: int, bias: bool = False):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.h = nheads
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=bias) for _ in range(4)
        ])

    def forward(
        self,
        inp: Tensor,
        count_bias: Tensor = None,
        attn_mask: Tensor = None,
        softcap: float = 0.0,
        past_kv_quantized: QuantizedKVCache = None,
        quantizer: TurboGeneQuantizer = None,
        layer_idx: int = 0,
        use_cache: bool = False,
        quantize_cache: bool = False,
    ):
        """
        Args:
            inp: (B, S_q, D)
            count_bias, attn_mask: attention bias and mask
            softcap: softcap value
            past_kv_quantized: quantized past KV cache
            quantizer: TurboGeneQuantizer instance
            layer_idx: layer index for quantizer
            use_cache: return KV cache
            quantize_cache: if True, quantize the KV cache
        """
        B = inp.size(0)
        q = self.linears[0](inp).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linears[1](inp).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linears[2](inp).view(B, -1, self.h, self.d_k).transpose(1, 2)

        # Prepend dequantized past KV if available
        if past_kv_quantized is not None and quantizer is not None:
            past_k = quantizer.dequantize_vector(
                past_kv_quantized.k_indices,
                past_kv_quantized.k_signs,
                past_kv_quantized.k_magnitudes,
                past_kv_quantized.k_norms,
                layer_idx, is_key=True,
            ).to(k.dtype)
            past_v = quantizer.dequantize_vector(
                past_kv_quantized.v_indices,
                past_kv_quantized.v_signs,
                past_kv_quantized.v_magnitudes,
                past_kv_quantized.v_norms,
                layer_idx, is_key=False,
            ).to(v.dtype)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Build quantized cache for current K/V
        present = None
        if use_cache and quantize_cache and quantizer is not None:
            k_idx, k_sgn, k_rmag, k_nrm = quantizer.quantize_vector(k, layer_idx, is_key=True)
            v_idx, v_sgn, v_rmag, v_nrm = quantizer.quantize_vector(v, layer_idx, is_key=False)
            present = QuantizedKVCache(
                k_indices=k_idx, v_indices=v_idx,
                k_centroids=quantizer.k_centroids[layer_idx],
                v_centroids=quantizer.v_centroids[layer_idx],
                k_signs=k_sgn, v_signs=v_sgn,
                k_magnitudes=k_rmag, v_magnitudes=v_rmag,
                k_norms=k_nrm, v_norms=v_nrm,
            )
        elif use_cache:
            present = (k, v)  # FP16 cache (no quantization)

        # Attention computation (always uses full-precision K/V)
        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if count_bias is not None:
            scores = scores + count_bias
        if softcap > 0:
            scores = torch.tanh(scores / softcap) * softcap
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        o = torch.matmul(weights, v)

        o = o.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linears[3](o), present
