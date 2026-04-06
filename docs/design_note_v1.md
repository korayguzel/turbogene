# TurboGene Design Note v1: Literatür Analizi ve Adaptasyon Soruları

**Tarih:** 2026-04-02  
**Faz:** 1 — Literatür + scGPT Analizi  
**Kaynaklar:** TurboQuant (arXiv:2504.19874), TurboESM (arXiv:2603.26110)

---

## 1. TurboQuant Temel Algoritması

### Stage 1: Random Orthogonal Rotation + Lloyd-Max Scalar Quantization

**Rotation (Π matrisi):**
- Π, i.i.d. Normal girişli bir matrisin QR decomposition'ından üretilen rastgele ortogonal matris
- `y = Π · x` dönüşümü uygulanır
- Rotation sonrası her koordinat yaklaşık Beta dağılımı izler; yüksek boyutlarda `N(0, 1/d)`'ye yakınsar
- Amaç: Outlier'ları tüm koordinatlara "yaymak" — worst-case koruması
- **Data-oblivious:** Kalibrasyon verisi yok, optimize edilmiyor

**Lloyd-Max Scalar Quantization:**
- Her koordinat bağımsız olarak b-bit ile quantize edilir
- [-1, 1] aralığını 2^b bucket'a böler
- Centroid'ler sürekli 1-D k-means çözerek bulunur:
  ```
  C(f_X, b) = min Σ ∫ |x - cᵢ|² · f_X(x) dx
  ```
- Rotation sonrası Beta dağılımı için bir kez hesaplanıp precompute edilir

**Quantize/Dequantize pipeline:**
1. `y = Π · x` (rotate)
2. Her `yⱼ` için en yakın centroid index'i bul → b-bit index
3. Dequant: centroid'leri al → `Π^T · ỹ` (rotate back)

**MSE sınırları (‖x‖=1 için):**

| Bit | D_mse üst sınır |
|-----|-----------------|
| 1   | 0.36            |
| 2   | 0.117           |
| 3   | 0.03            |
| 4   | 0.009           |

### Stage 2: QJL 1-Bit Residual Correction

**Problem:** MSE-optimized quantizer'lar inner product tahmininde biased (b=1'de 2/π bias).

**QJL tanımı:**
```
Q_qjl(x) = sign(S · x)         // S ∈ ℝ^{d×d}, i.i.d. N(0,1)
Q_qjl⁻¹(z) = (√(π/2)/d) · S^T · z   // z ∈ {-1,+1}^d
```

**İki aşamalı inner product quantizer:**
1. MSE quantizer'ı **(b-1) bit** ile uygula
2. Residual: `r = x - DeQuant_mse(Quant_mse(x))`
3. QJL: `sign(S · r)` → 1 bit/koordinat
4. Residual norm: `γ = ‖r‖₂` → 1 scalar FP

**Reconstruction:**
```
x̃ = x̃_mse + (√(π/2)/d) · γ · S^T · qjl_bits
```

**Kritik özellik — UNBIASED:** `E[⟨y, x̃⟩] = ⟨y, x⟩`

### Combined System

**Toplam bit bütçesi:** (b-1) bit MSE + 1 bit QJL = b bit/koordinat + 1 scalar norm

**Inner product distortion:**

| Bit | D_prod üst sınır |
|-----|-------------------|
| 1   | 1.57/d            |
| 2   | 0.56/d            |
| 3   | 0.18/d            |
| 4   | 0.047/d           |

**Compression:** ~4.5-5× (3.5-bit ≈ FP32 kalitesi)  
**Kalibrasyon:** Sıfır — Π rastgele, S rastgele, LUT precompute

---

## 2. TurboESM: ESM-2 Adaptasyonu

### 2.1 RoPE Çözümü

**Problem:** ESM-2 RoPE kullanıyor. Π, RoPE'dan önce uygulanırsa 2×2 boyut çiftleri bozulur.

**Çözüm:** RoPE önce, Π sonra. İspat:
```
(Π R_{θ,i} qᵢ)^T (Π R_{θ,j} kⱼ) = qᵢ^T R_{θ,i}^T Π^T Π R_{θ,j} kⱼ
                                    = qᵢ^T R_{θ,i}^T R_{θ,j} kⱼ    (Π^T Π = I)
```

**Pipeline:**
- Prefill: RoPE → full precision attention (hata=0) → Π → quantize → cache
- Decode: cache → dequantize → QJL correction → Π^T → attention

### 2.2 Protein Activation Outlier Profili

| Özellik | Tipik LLM | ESM-2 |
|---------|-----------|-------|
| Outlier/median ratio | 10-50× | 50-200× |
| Outlier kaynağı | Token embedding genel | Amino asit spesifik (sistein, katalitik triad) |
| Katman bağımlılığı | Kademeli artış | Geç katmanlarda (25-33) keskin artış |
| Vocabulary etkisi | 32K+ token | 20 amino asit, seyrek |

### 2.3 Domain-Specific Değişiklikler

**a) Head-Wise SVD Calibration:**
```
X_{l,h} = UΣV^T    →    Π_{l,h} = V^T
```
- TurboQuant'ın rastgele Π'si yerine, her layer×head için ayrı SVD
- Gerçek protein activation'larından calibrate

**b) Dual K/V LUT'ları:**
- lut_k: post-rotation key activation'larından
- lut_v: orijinal value activation'larından (V'ye rotation uygulanmıyor)
- Paylaşımlı LUT'a göre ~1.2 dB SNR kazancı

**c) QJL Adaptation:**
- Sign-only storage + per-head pre-calibrated mean absolute residual

### 2.4 Bileşen Transfer Durumu

| Bileşen | Domain-Agnostic | Protein-Specific |
|---------|----------------|-----------------|
| Lloyd-Max çerçevesi | ✅ | |
| Orthogonal rotation konsepti | ✅ | |
| QJL 1-bit residual prensibi | ✅ | |
| Rotation matrisi türetme | | ❌ SVD per-head |
| LUT calibration | | ❌ Protein veriden |
| Kalibrasyon veri seçimi | | ❌ Min 500 seq, SCOP |
| RoPE sıralama | | ❌ Mimari-specific |

### 2.5 Sonuçlar

- Compression: 330 MB → 47 MB (7.1×)
- Effective bit-width: 3.125 bit
- Prefill cosine similarity: 1.0000
- Decode cosine similarity: 0.9681 ortalama (0.9603–0.9757)
- Kalibrasyon: 2-5 dakika, ~200 MB checkpoint

---

## 3. scGPT Adaptasyon Soruları

### 3.1 Positional Encoding Etkileşimi

| Senaryo | Etki | Zorluk |
|---------|------|--------|
| RoPE | TurboESM çözümü direkt uygulanabilir | Düşük |
| Learned/absolute PE | Additive, Π ile çakışma yok | Düşük-Orta |
| Özel/hibrit PE | Analiz gerekli | Orta-Yüksek |

### 3.2 Gene Expression vs Protein Activation Farkları

| Boyut | Protein (ESM-2) | scGPT (Beklenti) |
|-------|-----------------|-------------------|
| Vocabulary | 20 amino asit | ~20,000+ gen + expression bin'leri |
| Outlier kaynağı | Yapısal motifler | Yüksek/düşük expressed genler |
| Değer aralığı | Discrete (20 kat.) | Continuous-binned expression |
| Sequence uzunluğu | 32-165 token | 100-2000+ gen |

### 3.3 Bileşen Transfer Planı

| Bileşen | Durum | scGPT Gereksinimi |
|---------|-------|-------------------|
| Lloyd-Max çerçevesi | ✅ Olduğu gibi | Centroid recalibration |
| QJL prensibi | ✅ Olduğu gibi | Per-head residual scale |
| Rotation konsepti | ✅ Olduğu gibi | SVD per-head calibration |
| Rotation türetme | ❌ Sıfırdan | scRNA-seq SVD |
| LUT calibration | ❌ Sıfırdan | Gene expression K/V LUT |
| PE uyumu | ⚠️ PE tipine bağlı | Faz 1'de belirlenecek |
| Triton kernel | ⚠️ Opsiyonel | Attention shape adaptasyonu |
| Kalibrasyon verisi | ❌ Sıfırdan | Çeşitli doku/hücre tipi |

### 3.4 Faz 1 Öncelikli Sorular

1. scGPT'nin PE tipi nedir?
2. KV cache VRAM'ın yüzde kaçını tutuyor? (kill criterion: %10)
3. Gene expression token'larının activation dağılımı nasıl?
4. scGPT head'leri ne kadar specialized?
5. Kalibrasyon için hangi scRNA-seq dataset'leri uygun?

---

## Sonraki Adım

scGPT repo'sunu klonlayıp şunları belirle:
1. PE mekanizması (forward() okuyarak)
2. Attention yapısı (head, head_dim, layer, causal masking)
3. Tokenization stratejisi (value binning, vocabulary, özel token'lar)
