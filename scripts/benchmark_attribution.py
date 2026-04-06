"""
TurboGene: Attribution Benchmark

Separates contributions of chunked attention vs KV quantization:
  1. Original: standard FlexAttention forward
  2. Chunked-only: chunked attention, NO quantization (FP16 K/V)
  3. Chunked+Quantized: chunked attention + pack/unpack/quantize K/V

For each: max batch size, throughput, peak VRAM.
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_all(checkpoint_dir):
    from scripts.test_sdpa_e2e import load_original_model
    from turbogene.sdpa_model import SDPATranscriptformer
    from turbogene.quantizer import TurboGeneQuantizer
    from turbogene.quantized_model import QuantizedTranscriptformer

    orig, vocab = load_original_model(checkpoint_dir)
    orig = orig.to("cuda", torch.float32)
    orig.eval()
    sdpa = SDPATranscriptformer(orig).to("cuda", torch.float32)
    sdpa.eval()

    mc = orig.model_config
    nh, hd = mc.num_heads, mc.embed_dim // mc.num_heads
    sl = mc.seq_len + mc.aux_len
    quantizer = TurboGeneQuantizer(mc.num_layers, nh, hd, n_bits=3).cuda()

    # Calibrate
    k_acts, v_acts = [], []
    def make_hook(li):
        def hook(mod, args):
            inp = args[0]
            B = inp.size(0)
            k_acts.append((li, mod.linears[1](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2).detach()))
            v_acts.append((li, mod.linears[2](inp).view(B, -1, mod.h, mod.d_k).transpose(1, 2).detach()))
        return hook
    hooks = [l.self_attn.register_forward_pre_hook(make_hook(i))
             for i, l in enumerate(sdpa.transformer_encoder.encoder_layers)]
    gen = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        sdpa(gene_token_indices=torch.randint(7, len(vocab), (2, sl), generator=gen).cuda(),
             gene_counts=torch.randint(1, 30, (2, sl), generator=gen).float().cuda())
    for h in hooks:
        h.remove()
    quantizer.calibrate(k_acts, v_acts)
    del k_acts, v_acts

    qmodel = QuantizedTranscriptformer(sdpa, quantizer).to("cuda", torch.float32)
    qmodel.eval()
    return orig, qmodel, vocab, mc


def measure(run_fn, batch_sizes, warmup=2, trials=3):
    results = {}
    for bs in batch_sizes:
        try:
            reset()
            for _ in range(warmup):
                run_fn(bs)
            torch.cuda.synchronize()

            reset()
            torch.cuda.reset_peak_memory_stats()
            run_fn(bs)
            torch.cuda.synchronize()
            vram = torch.cuda.max_memory_allocated() / 1024**2

            times = []
            for _ in range(trials):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_fn(bs)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            avg = np.mean(times)

            results[bs] = {"vram_mb": vram, "time_ms": avg * 1000, "cells_sec": bs / avg}
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results[bs] = "OOM"
                torch.cuda.empty_cache()
            else:
                raise
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {gpu_name} ({gpu_mb:.0f} MB)")

    print("\nLoading models...")
    orig, qmodel, vocab, mc = load_all(args.checkpoint_dir)
    sl = mc.seq_len + mc.aux_len

    gen = torch.Generator(device="cpu").manual_seed(42)
    g = torch.randint(7, len(vocab), (32, sl), generator=gen)
    c = torch.randint(1, 30, (32, sl), generator=gen).float()

    batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24]

    # ── 1. Original ──
    print("\n" + "=" * 70)
    print("1. Original (FlexAttention, no chunking, no quantization)")
    print("=" * 70)

    from transcriptformer.data.dataclasses import BatchData
    orig_hp = orig.half()

    def run_orig(bs):
        b = BatchData(gene_token_indices=g[:bs].cuda(),
                      gene_counts=c[:bs].cuda(), aux_token_indices=None)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            orig_hp(b, embed=True)

    r_orig = measure(run_orig, batch_sizes)
    for bs in batch_sizes:
        r = r_orig[bs]
        if r == "OOM":
            print(f"   batch={bs:2d}: OOM")
        else:
            print(f"   batch={bs:2d}: {r['vram_mb']:6.0f} MB, {r['time_ms']:7.1f} ms, {r['cells_sec']:5.1f} cells/sec")

    orig.float()
    del orig_hp
    reset()

    # ── 2. Chunked-only ──
    print("\n" + "=" * 70)
    print("2. Chunked-only (chunked attention, NO quantization)")
    print("=" * 70)

    qm_hp = qmodel.half()

    def run_chunked(bs):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            qm_hp.forward_chunked_only(g[:bs].cuda(), c[:bs].cuda(), embed=True)

    r_chunked = measure(run_chunked, batch_sizes)
    for bs in batch_sizes:
        r = r_chunked[bs]
        if r == "OOM":
            print(f"   batch={bs:2d}: OOM")
        else:
            print(f"   batch={bs:2d}: {r['vram_mb']:6.0f} MB, {r['time_ms']:7.1f} ms, {r['cells_sec']:5.1f} cells/sec")

    # ── 3. Chunked + Quantized ──
    print("\n" + "=" * 70)
    print("3. Chunked + Quantized (pack/unpack/dequantize)")
    print("=" * 70)

    def run_full(bs):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            qm_hp.forward_memory_efficient(g[:bs].cuda(), c[:bs].cuda(), embed=True)

    r_full = measure(run_full, batch_sizes)
    for bs in batch_sizes:
        r = r_full[bs]
        if r == "OOM":
            print(f"   batch={bs:2d}: OOM")
        else:
            print(f"   batch={bs:2d}: {r['vram_mb']:6.0f} MB, {r['time_ms']:7.1f} ms, {r['cells_sec']:5.1f} cells/sec")

    qmodel.float()
    del qm_hp
    reset()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("ATTRIBUTION SUMMARY")
    print("=" * 70)

    def max_batch(results):
        return max((bs for bs, r in results.items() if r != "OOM"), default=0)

    def best_tp(results):
        vals = [r["cells_sec"] for r in results.values() if r != "OOM"]
        return max(vals) if vals else 0

    mb_orig = max_batch(r_orig)
    mb_chunked = max_batch(r_chunked)
    mb_full = max_batch(r_full)

    tp_orig = best_tp(r_orig)
    tp_chunked = best_tp(r_chunked)
    tp_full = best_tp(r_full)

    print(f"\n   {'Method':40s} {'Max Batch':>10s} {'Best cells/sec':>15s}")
    print(f"   {'-'*65}")
    print(f"   {'Original (FlexAttention)':40s} {mb_orig:>10d} {tp_orig:>15.1f}")
    print(f"   {'Chunked-only (no quantization)':40s} {mb_chunked:>10d} {tp_chunked:>15.1f}")
    print(f"   {'Chunked + Quantized (TurboGene)':40s} {mb_full:>10d} {tp_full:>15.1f}")

    if mb_orig > 0:
        print(f"\n   Batch scaling from chunked attention:  {mb_orig} -> {mb_chunked} ({mb_chunked/mb_orig:.1f}x)")
        if mb_chunked > 0:
            print(f"   Additional scaling from quantization:  {mb_chunked} -> {mb_full} ({mb_full/mb_chunked:.1f}x)")
        print(f"   Total scaling:                         {mb_orig} -> {mb_full} ({mb_full/mb_orig:.1f}x)")

    if tp_orig > 0:
        print(f"\n   Throughput from chunked attention:  {tp_orig:.1f} -> {tp_chunked:.1f} ({tp_chunked/tp_orig:.2f}x)")
    if tp_chunked > 0:
        print(f"   Quantization overhead:              {tp_chunked:.1f} -> {tp_full:.1f} ({(1-tp_full/tp_chunked)*100:+.1f}%)")

    if 4 in r_orig and r_orig[4] != "OOM":
        print(f"\n   VRAM at batch=4:")
        print(f"     Original:          {r_orig[4]['vram_mb']:.0f} MB")
        if 4 in r_chunked and r_chunked[4] != "OOM":
            print(f"     Chunked-only:      {r_chunked[4]['vram_mb']:.0f} MB")
        if 4 in r_full and r_full[4] != "OOM":
            print(f"     Chunked+Quantized: {r_full[4]['vram_mb']:.0f} MB")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
