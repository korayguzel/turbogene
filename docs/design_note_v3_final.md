# TurboGene Final Results Summary

> **Note (2026-04-05):** Numbers in this document reflect the pre-fix implementation. After PolarQuant norm extraction and proper QJL, key metrics changed: throughput 2.79x → 1.79x, embedding cosine 1.000 → 0.981, compression 3.76x → 3.66x. See paper/main.tex for current numbers.

**Date:** 2026-04-02  
**Hardware:** NVIDIA RTX 4070 (12 GB VRAM)

---

## Paper-Ready Results Table

### Table 1: VRAM and Batch Size
| Batch | Original | TurboGene | Status |
|-------|---------|-----------|--------|
| 1 | 2,005 MB | 1,446 MB | Both fit |
| 4 | 5,534 MB | 2,160 MB | Both fit |
| 8 | OOM | **3,113 MB** | **TurboGene only** |
| 12 | OOM | **4,067 MB** | **TurboGene only** |
| 16 | OOM | **5,019 MB** | **TurboGene only** |

### Table 2: Cell Type Classification (1K cells, 9 types, 5-fold CV)
| Method | Accuracy | F1 | ARI |
|--------|----------|-----|-----|
| Original (batch=4) | 0.978 | 0.971 | 0.958 |
| TurboGene (batch=16) | 0.978 | 0.971 | 0.958 |
| Delta | 0.000 | 0.000 | 0.000 |

### Table 3: Throughput
| Method | Batch | Cells/sec | Time for 10K cells |
|--------|-------|-----------|-------------------|
| Original | 4 | 3.9 | 43 min |
| **TurboGene** | **16** | **8.4** | **20 min** |
| Speedup | — | **2.18x** | **2.18x** |

### Table 4: KV Cache Quantization Comparison
| Method | Bits | Compress | Attn cos_sim | Min cos_sim |
|--------|------|---------|-------------|-------------|
| FP16 | 16.0 | 1.00x | 1.000 | 1.000 |
| KIVI 2-bit | 2.3 | 7.06x | 0.917 | 0.799 |
| **TurboGene** | **4.3** | **3.76x** | **0.990** | **0.969** |

### Table 5: Ablation
| Config | Attn cos_sim |
|--------|-------------|
| Rotation only | 1.000 |
| + Lloyd-Max 3-bit | 0.976 |
| + QJL 1-bit (full) | **0.990** |
| Random rotation | **0.990** |
| SVD rotation | 0.888 |
| Shared LUT | 0.976 |
| Dual K/V LUT | **0.990** |

### Table 6: Weight-Only INT8 Baseline
| Method | Model Size | Quality (mu cos) |
|--------|-----------|-----------------|
| FP32 | 1,640 MB | 1.000 |
| INT8 weight | 234 MB (7.0x) | 0.758 |
| TurboGene KV | 820 MB + 204 MB KV | 1.000 (prefill) / 0.990 (decode) |

Weight quant and KV quant are complementary — weight quant reduces model memory, KV quant reduces attention memory.

### Embedding Similarity
| Dataset | Cells | cos_sim |
|---------|-------|---------|
| Human cardiac (1K) | 1,000 | 1.000000 |

---

## Key Claims for Paper

1. **4x batch scaling**: batch 4 -> 16 on RTX 4070 (12 GB)
2. **2.18x throughput**: 10K cells in 20 min vs 43 min
3. **Zero quality loss**: classification Acc/F1/ARI identical, embedding cos=1.0
4. **TurboGene > KIVI**: 0.990 vs 0.917 attention cos_sim, all 12 layers
5. **3.76x KV compression**: 768 MB -> 204 MB (theoretical)
6. **First KV cache quant for single-cell transformers**
