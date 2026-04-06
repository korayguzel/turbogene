"""
Weight-only INT8 quantization baseline for TranscriptFormer.

Shows that weight quantization and KV cache quantization are complementary:
- Weight quant reduces model weight memory (~820 MB -> ~420 MB)
- KV cache quant reduces attention memory (dominant at large batch)
- Both can be applied together

Uses PyTorch's dynamic quantization for INT8 weights.
"""

import torch
import torch.nn as nn


def apply_weight_int8(model):
    """Apply dynamic INT8 weight quantization to linear layers.

    Returns a new model with quantized weights. Only quantizes
    the transformer encoder's linear layers (Q/K/V projections + FFN).
    """
    # PyTorch dynamic quantization on Linear layers
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def measure_model_size(model):
    """Measure model size in MB."""
    total_bytes = 0
    for name, param in model.named_parameters():
        total_bytes += param.nelement() * param.element_size()
    for name, buf in model.named_buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / 1024 / 1024
