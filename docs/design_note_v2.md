# TurboGene Design Note v2: TranscriptFormer Mimari Analizi

**Tarih:** 2026-04-02  
**Faz:** 1 — TranscriptFormer Analizi (scGPT'den Pivot)  
**Kaynak:** TranscriptFormer repo (github.com/czi-ai/transcriptformer), bioRxiv 2025.04.25.650731v2

---

## Pivot Kararı: scGPT → TranscriptFormer

### scGPT Neden Kapsam Dışına Çıktı

scGPT repo analizi (2026-04-02) kesin olarak gösterdi ki:
- scGPT **encoder-only** mimari kullanıyor (TransformerEncoder)
- Training paradigması **Masked Language Model (MLM)** — BERT tarzı
- `generate_square_subsequent_mask()` fonksiyonu tanımlı ama **hiçbir yerde çağrılmıyor**
- `generate()` metodu iteratif refinement, token-by-token autoregressive decode değil
- **KV cache kullanmıyor** — her forward pass'ta tüm attention yeniden hesaplanıyor

Sonuç: KV cache quantization scGPT'de anlamlı bir etki yaratmaz.

### TranscriptFormer Neden Yeni Ana Hedef

- **Gerçek autoregressive**: causal mask + right-shift, kodda doğrulandı
- **KV cache bottleneck**: 201 MB/sample (FP16), batch=16'da 3.2 GB
- **Positional encoding yok**: TurboQuant rotation ile çakışma riski sıfır
- **RTX 4070'e sığar**: TF-Sapiens batch=8, ~7 GB tahmini
- **Single-cell domain**: KV cache quantization ilk defa uygulanacak — net novelty

---

## 1. Positional Encoding: YOK

TranscriptFormer hiçbir positional encoding kullanmıyor:
- RoPE yok, sinusoidal yok, learned PE yok
- Gene sırası her batch'te shuffle ediliyor (randomize_genes=True)
- Causal mask autoregressive yapıyı sağlıyor ama sıra-bağımsız

**TurboQuant implikasyonu:** En iyi senaryo. TurboESM'deki RoPE-Π problemi tamamen ortadan kalkıyor.

---

## 2. Attention Yapısı

### Mimari Parametreleri (Tüm variant'lar aynı)

| Parametre | Değer |
|-----------|-------|
| embed_dim | 2048 |
| model_dim (FFN) | 2048 |
| num_heads | 16 |
| head_dim | 128 (= 2048/16) |
| num_layers | 12 |
| seq_len | 2047 (+1 aux = 2048 total) |
| block_len | 128 (Flex Attention block size) |
| activation | GELU |
| pre_norm | Evet |
| softcap | 10 |
| bias | Hayır (attn + FFN) |

### Variant Farkları (Sadece vocabulary + frozen embeddings)

| Variant | Trainable | Non-trainable (frozen) | Vocab Size | Training Cells |
|---------|-----------|----------------------|------------|----------------|
| TF-Sapiens | 368M | 61M | 23,829 | 57M human |
| TF-Exemplar | 542M | 282M | 110,290 | 110M (5 species) |
| TF-Metazoa | 444M | 633M | 247,388 | 112M (12 species) |

### FlexAttention + Expression-Aware Score Mod

TranscriptFormer PyTorch'un `flex_attention` API'sini kullanıyor:

```python
# Causal mask: standart lower-triangular
causal_mask(b, h, q_idx, kv_idx) = (q_idx >= kv_idx)

# Expression-aware score modification:
score_mod(score, b, h, q_idx, kv_idx):
    bias = log1p(count[b, kv_idx]) * (q_idx > kv_idx)
    score = score + bias
    score = tanh(score / softcap) * softcap  # logit capping
```

Yüksek expression'lı genler daha yüksek attention alıyor.

---

## 3. Autoregressive Mekanizma

### Right-Shift + Teacher Forcing

```
Input:  [START], gene_1, gene_2, ..., gene_{N-1}  (right-shifted)
Target: gene_1,  gene_2, gene_3, ..., gene_N, [END]  (gene ID prediction)
Counts: count_1, count_2, ..., count_N             (ZTP count prediction)
```

### Dual-Head Prediction

1. **Count Head** (CountDecoderHead):
   - Input: [transformer_output; gene_embedding] (skip connection, 2×embed_dim)
   - Output: μ (Poisson rate)
   - Loss: Zero-Truncated Poisson NLL
   
2. **Gene ID Head** (MLP → vocab_size):
   - Input: transformer output
   - Output: next gene logits
   - Loss: CrossEntropyLoss

### Zero-Truncated Poisson Count Model

```
NLL = -count * log(μ + ε) + log(exp(μ) - 1) + log(Γ(count + 1))
```

---

## 4. Tokenization ve Embedding

### Gene Token'ları
- Her gen Ensembl ID ile temsil
- Gene embedding'leri **ESM-2 protein embedding'lerinden** türetiliyor (frozen)
- ESM-2 embedding → 2-layer MLP → 2048 dim
- Freeze=True: embedding'ler training'de güncellenmez

### Expression Values
- Raw count'lar (integer) → clip_counts=30 ile kırpılır
- Count bilgisi attention score'a log1p bias olarak giriyor
- Ayrı value encoder yok

### Özel Token'lar
| Token | Amaç |
|-------|-------|
| [START] | Autoregressive başlangıç |
| [END] | Dizi sonu |
| [PAD] | Padding |
| [CELL] | Hücre-level token |
| [MASK] | Maskeleme |
| unknown | Bilinmeyen gen |

---

## 5. RTX 4070 (12 GB) Fizibilite

### Model Weight Boyutu (TF-Sapiens, FP16)

| Bileşen | Parametre | FP16 (MB) |
|---------|-----------|-----------|
| Gene embeddings (frozen) | ~61M | ~122 |
| Transformer (12L×16H×2048) | ~368M | ~736 |
| Count head + Gene ID head | ~10-20M | ~30 |
| **Toplam** | **~429M** | **~888** |

### VRAM Tahmini (Inference, FP16)

| Senaryo | VRAM Tahmini | RTX 4070 |
|---------|-------------|----------|
| batch=1 | ~1.6 GB | Rahat sığar |
| batch=4 | ~3.9 GB | Rahat sığar |
| batch=8 | ~6.9 GB | Sığar |
| batch=16 | ~12-13 GB | Sınırda |
| batch=32 | ~22 GB | Sığmaz |

### KV Cache Boyutu

```
KV cache per sample = 2 × 12 layers × 16 heads × 128 head_dim × 2048 seq × 2 bytes
                    = 201 MB (FP16)
```

| Batch | FP16 KV Cache | TurboQuant 3-bit KV | Tasarruf |
|-------|--------------|--------------------|---------| 
| 1 | 201 MB | ~38 MB | 163 MB |
| 4 | 804 MB | ~150 MB | 654 MB |
| 8 | 1,608 MB | ~301 MB | 1,307 MB |
| 16 | 3,216 MB | ~602 MB | 2,614 MB |

**TurboQuant ile batch=16 mümkün hale geliyor** (12 GB VRAM sınırı içinde).

---

## 6. scGPT vs TranscriptFormer Karşılaştırma

| Özellik | scGPT | TranscriptFormer |
|---------|-------|-----------------|
| Mimari | Encoder-only (MLM) | Decoder-style (causal) |
| Autoregressive | Hayır | **Evet** |
| KV Cache | Yok | **201 MB/sample** |
| PE | Yok | Yok |
| embed_dim | 512 | **2048** |
| head_dim | 64 | **128** |
| num_heads × layers | 8×12 | **16×12** |
| Trainable params | 53M | **368M-542M** |
| Value encoding | MLP veya binned embedding | **Attention score bias** |
| Gene embedding | nn.Embedding (trained) | **ESM-2 frozen → MLP** |
| Count model | MSE loss | **ZTP Poisson NLL** |
| RTX 4070 fit | Rahat | Sığar (batch≤8) |

---

## 7. Downstream Tasks

| Task | Decode-Heavy? | Detay |
|------|--------------|-------|
| Cell embedding extraction | Hayır | Mean-pooled cell embedding |
| Cell type classification | Hayır | Zero-shot cross-species kNN |
| Disease state identification | Hayır | Human hastalık tespiti |
| **Log-likelihood evaluation** | **Evet** | ZTP NLL + Gene ID CE |
| Gene regulatory network | Kısmen | Contextual gene embedding'ler |
| Cross-species transfer | Hayır | OOD species |
| **Generative sampling** | **Evet** | ZTP sample + CE sample |

---

## 8. TurboQuant Adaptasyon Planı (Ön Değerlendirme)

### Olduğu Gibi Alınabilir

| Bileşen | Neden? |
|---------|--------|
| Lloyd-Max quantization çerçevesi | Domain-agnostic |
| QJL 1-bit residual correction | Domain-agnostic |
| Orthogonal rotation konsepti | PE yok → çakışma yok |

### Domain-Specific Adaptasyon Gerekli

| Bileşen | Neden? |
|---------|--------|
| Rotation matrisi (Π) | Head-wise SVD, scRNA-seq activation'larından calibrate |
| K/V LUT'ları | Gene expression activation dağılımına özel |
| Score_mod uyumu | log-count bias + quantized attention interaction |
| Kalibrasyon verisi | Çeşitli hücre tipleri/dokulardan scRNA-seq |

### Özel Zorluklar

1. **FlexAttention → KV cache**: flex_attention KV cache desteklemiyor, standart SDPA'ya geçiş gerekebilir
2. **Score_mod quantization etkisi**: log-count bias quantized K ile nasıl etkileşir?
3. **ESM-2 frozen embedding**: Quantization gene embedding'leri etkilemez (frozen + MLP projection)

---

## Sonraki Adımlar

1. TF-Sapiens checkpoint'u indir, RTX 4070'te gerçek VRAM profiling
2. K/V activation dağılımlarını çıkar (head-wise histogram + outlier analizi)
3. flex_attention → SDPA + KV cache dönüşüm fizibilitesi değerlendir
4. Kalibrasyon veri seti seçimi (CellxGene'den çeşitli doku/hücre tipi)
