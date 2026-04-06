# TurboGene

**Memory-Efficient Inference for TranscriptFormer on Consumer GPUs**

TurboGene converts [TranscriptFormer](https://github.com/czi-ai/transcriptformer) from FlexAttention to standard SDPA and adds chunked attention, enabling **6x larger batch sizes** and **3x throughput** on consumer GPUs. An experimental KV cache quantization module is also included.

## Key Results (RTX 4070, 12 GB VRAM)

| Metric | Original | Chunked-only | Chunked + Quantized |
|--------|----------|-------------|---------------------|
| Max batch size | 4 | **24** | 16 |
| Best throughput | 3.8 cells/sec | **11.1 cells/sec** | 5.7 cells/sec |
| VRAM @ batch=4 | 6,697 MB | **2,837 MB** | 3,259 MB |
| Cell type accuracy | 0.979 | 0.979 | 0.979 |
| Embedding cosine | -- | 1.000 | 0.981 |

**What each component contributes:**
- **SDPA + chunked attention** (the main contribution): batch 4 → 24 (6x), throughput 3.8 → 11.1 cells/sec (2.9x), VRAM 6.7 GB → 2.8 GB at batch=4
- **KV cache quantization** (experimental): reduces max batch from 24 to 16 due to pack/unpack overhead, but adds decode-quality simulation (embedding cosine 0.981, useful for studying quantization effects on biological embeddings)

## What TurboGene Does

TurboGene is an **inference-time optimization**. It does not modify model training or weights.

1. **SDPA conversion**: Converts TranscriptFormer's FlexAttention (which doesn't support KV caching) to standard scaled dot-product attention with explicit bias matrices. Preserves softcap-mask ordering for correctness. Output is bit-identical to the original (cosine similarity = 1.000000).

2. **Chunked attention**: Splits the query sequence into 256-position chunks, each attending only to its causal context. Reduces peak attention memory from O(BHS^2) to O(BHS*C) where C=256. This is the primary source of memory and throughput improvement.

3. **KV cache quantization** (experimental): PolarQuant norm extraction + orthogonal rotation + Lloyd-Max 3-bit scalar quantization + QJL 1-bit residual correction via Johnson-Lindenstrauss random projection. Adapted from [TurboQuant](https://arxiv.org/abs/2504.19874) with reference implementation from [turboquant_plus](https://github.com/TheTom/turboquant_plus) (Apache 2.0). Currently adds computational overhead that reduces batch capacity; useful for studying quantization effects on biological embeddings rather than for production memory savings.

This is complementary to traditional approaches like PCA-based pipelines (Scanpy/Seurat); TurboGene is relevant when using foundation model embeddings for tasks that PCA cannot support, such as zero-shot cross-species classification, perturbation prediction, and generative cell profiling.

## Installation

**Requirements:** Python >= 3.11, CUDA GPU (tested on RTX 4070 12 GB)

```bash
git clone https://github.com/korayguzel/turbogene.git
cd turbogene
pip install -e .
```

### Download Model Checkpoint

```bash
pip install transcriptformer
python -c "
from transcriptformer.utils import download_checkpoint
download_checkpoint('tf-sapiens', 'checkpoints/tf_sapiens')
"
```

### Download Benchmark Datasets

```bash
python scripts/download_datasets.py
```

## Quick Start

```bash
python examples/quickstart.py \
  --data your_cells.h5ad \
  --checkpoint-dir checkpoints/tf_sapiens \
  --batch-size 24 \
  --output embeddings.h5ad
```

The input `.h5ad` file should contain raw counts with Ensembl gene IDs in `adata.var`.

## Benchmarks

```bash
CKPT=checkpoints/tf_sapiens

# SDPA conversion validation (cosine = 1.000000)
python scripts/test_sdpa_e2e.py --checkpoint-dir $CKPT

# Attribution benchmark: Original vs Chunked-only vs Chunked+Quantized
python scripts/benchmark_attribution.py --checkpoint-dir $CKPT

# Quantizer unit tests
python scripts/test_quantizer.py --checkpoint-dir $CKPT

# Full biology benchmark (10K CellxGene cells)
python scripts/final_benchmark.py --checkpoint-dir $CKPT

# Ablation study (quantization components)
python scripts/ablation_study.py --checkpoint-dir $CKPT
```

## Project Structure

```
turbogene/
  sdpa_layers.py        # SDPA attention with softcap, KV cache, bias/mask
  sdpa_model.py         # SDPATranscriptformer wrapper (main contribution)
  quantized_model.py    # QuantizedTranscriptformer with chunked + quantized modes
  quantizer.py          # KV cache quantizer: norm extraction, rotation, Lloyd-Max, QJL
  data_utils.py         # Fast vectorized tokenization for H5AD
  baselines.py          # KIVI 2-bit baseline
scripts/                # Benchmarks and ablations
examples/               # User-facing quickstart
docs/                   # Design notes
```

## License

MIT
