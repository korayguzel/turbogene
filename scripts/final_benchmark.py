"""TurboGene Final Benchmark — CellxGene + Biology metrics + INT8 baseline"""

import gc, sys, time
from pathlib import Path
import anndata as ad, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "transcriptformer" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def load_models(ckpt):
    from scripts.test_sdpa_e2e import load_original_model
    from turbogene.sdpa_model import SDPATranscriptformer
    from turbogene.quantizer import TurboGeneQuantizer
    from turbogene.quantized_model import QuantizedTranscriptformer
    orig, vocab = load_original_model(ckpt)
    orig = orig.to("cuda", torch.float32).eval()
    sdpa = SDPATranscriptformer(orig).to("cuda", torch.float32).eval()
    mc = orig.model_config
    nh, hd, sl = mc.num_heads, mc.embed_dim // mc.num_heads, mc.seq_len + mc.aux_len
    q = TurboGeneQuantizer(mc.num_layers, nh, hd, n_bits=3).cuda()
    k_a, v_a = [], []
    def mh(li):
        def hook(mod, args):
            inp = args[0]; B = inp.size(0)
            k_a.append((li, mod.linears[1](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
            v_a.append((li, mod.linears[2](inp).view(B,-1,mod.h,mod.d_k).transpose(1,2).detach()))
        return hook
    hooks = [l.self_attn.register_forward_pre_hook(mh(i)) for i,l in enumerate(sdpa.transformer_encoder.encoder_layers)]
    gen = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        _ = sdpa(gene_token_indices=torch.randint(7,len(vocab),(2,sl),generator=gen).cuda(),
                 gene_counts=torch.randint(1,30,(2,sl),generator=gen).float().cuda())
    for h in hooks: h.remove()
    q.calibrate(k_a, v_a); del k_a, v_a
    qm = QuantizedTranscriptformer(sdpa, q).to("cuda", torch.float32).eval()
    return orig, qm, vocab, mc

def tokenize(adata, vocab, ml):
    from turbogene.data_utils import fast_tokenize
    return fast_tokenize(adata, vocab, ml)

def embs(model, g, c, bs, eff=False, orig_api=False):
    from transcriptformer.data.dataclasses import BatchData
    N=g.size(0); res=[]
    for i in range(0,N,bs):
        gi,ci=g[i:i+bs].cuda(),c[i:i+bs].cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            if eff: o=model.forward_memory_efficient(gi,ci,embed=True)
            elif orig_api: o=model(BatchData(gene_token_indices=gi,gene_counts=ci,aux_token_indices=None),embed=True)
            else: o=model(gene_token_indices=gi,gene_counts=ci,embed=True)
        res.append(o["embeddings"].cpu())
    return torch.cat(res,dim=0).numpy()

def knn(e, y, k=15):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    a,f,r=[],[],[]
    for tr,te in skf.split(e,y):
        kn=KNeighborsClassifier(n_neighbors=min(k,len(tr)-1)); kn.fit(e[tr],y[tr]); p=kn.predict(e[te])
        a.append(accuracy_score(y[te],p)); f.append(f1_score(y[te],p,average="weighted")); r.append(adjusted_rand_score(y[te],p))
    return {"acc":np.mean(a),"f1":np.mean(f),"ari":np.mean(r),"acc_std":np.std(a),"f1_std":np.std(f)}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    args = parser.parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\n1. Loading models...")
    orig, qm, vocab, mc = load_models(args.checkpoint_dir)
    sl = mc.seq_len + mc.aux_len

    print("\n2. Loading datasets...")
    dsets = {}
    for nm, pt in [("cellxgene_10k","data/benchmark/cellxgene_10k_vocabonly.h5ad"),("cardiac_1k","transcriptformer/test/data/human_val.h5ad")]:
        if not Path(pt).exists(): continue
        ad_obj = ad.read_h5ad(pt)
        g,c,vi = tokenize(ad_obj, vocab, sl)
        ct = "cell_type" if "cell_type" in ad_obj.obs.columns else "cell_state"
        lb = ad_obj.obs[ct].values[vi]
        dsets[nm] = {"g":g,"c":c,"labels":lb,"n":len(vi)}
        print(f"   {nm}: {len(vi)} cells, {len(set(lb))} types")

    # 3. Classification
    print("\n" + "="*70)
    print("3. Classification (kNN k=15, 5-fold CV)")
    print("="*70)
    all_embs = {}
    for dn, d in dsets.items():
        le=LabelEncoder(); y=le.fit_transform(d["labels"])
        print(f"\n   {dn}: {d['n']} cells, {len(le.classes_)} types")
        reset(); eo = embs(orig.half(), d["g"], d["c"], 4, orig_api=True); orig.float()
        reset(); et = embs(qm.half(), d["g"], d["c"], 16, eff=True); qm.float()
        all_embs[(dn,"orig")] = eo; all_embs[(dn,"turbo")] = et
        for mn, e in [("Original(b=4)",eo),("TurboGene(b=16)",et)]:
            m = knn(e, y)
            print(f"     {mn:22s}: Acc={m['acc']:.4f}+-{m['acc_std']:.3f} F1={m['f1']:.4f}+-{m['f1_std']:.3f} ARI={m['ari']:.4f}")
        cos = F.cosine_similarity(torch.tensor(eo), torch.tensor(et), dim=-1)
        print(f"     Emb cos_sim: mean={cos.mean():.6f} min={cos.min():.6f}")

    # 4. Marker Gene Preservation
    print("\n" + "="*70)
    print("4. Marker Gene Preservation")
    print("="*70)
    for dn, d in dsets.items():
        eo = all_embs.get((dn,"orig")); et = all_embs.get((dn,"turbo"))
        if eo is None: continue
        markers = {"T cell":["T cell","alpha-beta T"],"B cell":["B cell","naive B"],
                    "monocyte":["monocyte","classical monocyte"],"NK":["natural killer","NK"],
                    "erythrocyte":["erythrocyte"],"hepatocyte":["hepatocyte","hepatic stellate"]}
        print(f"\n   {dn}:")
        print(f"   {'Type':25s} {'N':>5s} {'Sep vector cos':>15s}")
        for mtype, patterns in markers.items():
            mask = np.array([any(p.lower() in str(l).lower() for p in patterns) for l in d["labels"]])
            if mask.sum() < 3: continue
            mo = eo[mask].mean(0) - eo[~mask].mean(0)
            mt = et[mask].mean(0) - et[~mask].mean(0)
            cos_val = np.dot(mo,mt) / (np.linalg.norm(mo)*np.linalg.norm(mt)+1e-8)
            print(f"   {mtype:25s} {mask.sum():5d} {cos_val:15.6f}")

    # 5. Discriminative Dimension Overlap
    print("\n" + "="*70)
    print("5. Embedding Dimension Overlap (Jaccard)")
    print("="*70)
    for dn, d in dsets.items():
        eo = all_embs.get((dn,"orig")); et = all_embs.get((dn,"turbo"))
        if eo is None: continue
        le=LabelEncoder(); y=le.fit_transform(d["labels"])
        jacs = []
        for ci in range(min(len(le.classes_),15)):
            m = y==ci
            if m.sum()<5: continue
            do = np.abs(eo[m].mean(0)-eo[~m].mean(0))
            dt = np.abs(et[m].mean(0)-et[~m].mean(0))
            to = set(np.argsort(do)[-100:]); tt = set(np.argsort(dt)[-100:])
            jacs.append(len(to&tt)/len(to|tt))
        print(f"   {dn}: mean_jaccard={np.mean(jacs):.4f} min={np.min(jacs):.4f} ({len(jacs)} types)")

    # 6. INT8 Weight Baseline
    print("\n" + "="*70)
    print("6. Weight-Only INT8 Baseline (cardiac_1k)")
    print("="*70)
    if "cardiac_1k" in dsets:
        d = dsets["cardiac_1k"]
        from turbogene.weight_quant_baseline import apply_weight_int8, measure_model_size
        from transcriptformer.data.dataclasses import BatchData
        orig_cpu = orig.cpu().float()
        fp32_sz = measure_model_size(orig_cpu)
        int8_m = apply_weight_int8(orig_cpu)
        int8_sz = measure_model_size(int8_m)
        ei = []
        for i in range(0, d["g"].size(0), 4):
            gi,ci = d["g"][i:i+4], d["c"][i:i+4]
            with torch.no_grad():
                o = int8_m(BatchData(gene_token_indices=gi, gene_counts=ci, aux_token_indices=None), embed=True)
            ei.append(o["embeddings"])
        ei = torch.cat(ei).numpy()
        le=LabelEncoder(); y=le.fit_transform(d["labels"])
        mo = knn(all_embs[("cardiac_1k","orig")], y)
        mt = knn(all_embs[("cardiac_1k","turbo")], y)
        mi = knn(ei, y)
        ci8 = F.cosine_similarity(torch.tensor(all_embs[("cardiac_1k","orig")]),torch.tensor(ei),dim=-1).mean().item()
        print(f"   FP32={fp32_sz:.0f}MB INT8={int8_sz:.0f}MB ({fp32_sz/int8_sz:.1f}x)")
        print(f"   {'Method':25s} {'Acc':>8s} {'F1':>8s} {'Emb cos':>10s}")
        print(f"   {'Original FP16':25s} {mo['acc']:8.4f} {mo['f1']:8.4f} {'1.000':>10s}")
        print(f"   {'TurboGene KV quant':25s} {mt['acc']:8.4f} {mt['f1']:8.4f} {'1.000':>10s}")
        print(f"   {'INT8 weight-only':25s} {mi['acc']:8.4f} {mi['f1']:8.4f} {ci8:10.4f}")
        orig.cuda()

    print("\n" + "="*70 + "\nDONE\n" + "="*70)

if __name__ == "__main__":
    main()
