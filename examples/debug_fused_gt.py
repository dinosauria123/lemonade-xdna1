#!/usr/bin/env python3
"""DECISIVE (final, overflow-inclusive): fused-NPU vs torch SDPA ground truth,
with max|T| (overflow magnitude) per call.

Hook the REAL fused kernel, capture (Q,K,V,fused_out,max|T|) per call. Then
compute torch SDPA ground truth from the SAVED real Q/K/V and compare per call.

This is the decisive test: it isolates the KERNEL's truth against torch SDPA
without any -30-vs-inf convention contamination (uses torch SDPA, not a -30
numpy ref), AND reports max|T| so we can see whether overflow is the cause.
"""
import os, sys
os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
sys.path.insert(0, "/home/dino/open-xdna/server")
import numpy as np, torch
import torch.nn.functional as F
import importlib, engine as engine_mod
import npu_gemm


def torch_sdpa_gt(Qh, Kh, Vh, scale, n_rep, causal, D):
    n_hq, S_q = Qh.shape[0], Qh.shape[1]
    n_hkv, S_k = Kh.shape[0], Kh.shape[1]
    Qb = Qh.reshape(1, n_hq, S_q, D)
    Kb = Kh.reshape(1, n_hkv, S_k, D)
    Vb = Vh.reshape(1, n_hkv, S_k, D)
    out = np.zeros((1, n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        qg = torch.as_tensor(Qb[:, g * n_rep:(g + 1) * n_rep])
        kg = torch.as_tensor(Kb[0, g]).unsqueeze(0).expand(1, n_rep, S_k, D)
        vg = torch.as_tensor(Vb[0, g]).unsqueeze(0).expand(1, n_rep, S_k, D)
        attn = None
        if causal and S_q > 1:
            attn = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool), 1)
        o = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=attn)
        out[:, g * n_rep:(g + 1) * n_rep] = o.numpy()
    return out


def main():
    importlib.reload(engine_mod)
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    cfg = e.model.config
    n_hq = int(cfg.num_attention_heads); n_hkv = int(cfg.num_key_value_heads)
    n_rep = n_hq // n_hkv
    D = int(cfg.hidden_size // n_hq)
    print(f"heads={n_hq} kv={n_hkv} n_rep={n_rep} D={D} layers={len(e.layers)} "
          f"npu_avail={e.npu_available}\n", flush=True)

    calls = []
    orig_fused = npu_gemm.npu_attention_fused_impl
    def hook_fused(Q, K, V, scale, n_rep=1, causal=False):
        r = orig_fused(Q, K, V, scale, n_rep, causal)
        Qs = Q.astype(np.float32) * float(scale)
        T = np.einsum("rqd,kid->rqk", Qs, K.astype(np.float32))
        calls.append(dict(Q=Q.astype(np.float32).copy(),
                          K=K.astype(np.float32).copy(),
                          V=V.astype(np.float32).copy(),
                          out=r.copy(), causal=bool(causal),
                          scale=float(scale), D=int(Q.shape[-1]),
                          n_rep=int(n_rep), maxT=float(np.abs(T).max())))
        return r
    npu_gemm.npu_attention_fused_impl = hook_fused

    list(e.generate([{"role": "user", "content": "What is 2+2? Answer in one word."}],
                    {"max_tokens": 4, "temperature": 0.0, "top_k": 1}))
    npu_gemm.npu_attention_fused_impl = orig_fused
    print(f"fused calls = {len(calls)}\n", flush=True)

    worst = 0.0; wc = 0; rows = []
    for i, c in enumerate(calls):
        gt = torch_sdpa_gt(c["Q"], c["K"], c["V"], c["scale"], c["n_rep"],
                           c["causal"], c["D"])
        f = c["out"]
        S_q = c["Q"].shape[1]
        d = float(np.abs(f - gt[0]).max())
        denom = float(np.abs(gt).max()) + 1e-6
        rd = d / denom
        rows.append((i, S_q, c["causal"], d, rd, c["maxT"],
                     float(np.abs(c["Q"]).max()), float(np.abs(c["K"]).max())))
        if rd > worst: worst, wc = rd, i
    rows.sort(key=lambda r: -r[4])
    print("--- fused-NPU vs torch SDPA ground truth (per call) ---")
    for r in rows[:12]:
        print(f"  call{r[0]} S_q={r[1]:2d} caus={r[2]} rel={r[4]:.6f} "
              f"dmax={r[3]:.5f} max|T|={r[5]:7.1f} |Q|={r[6]:.1f} |K|={r[7]:.1f}")
    print(f"\nWORST RELATIVE DMAX = {worst:.6f} at call {wc}")
    if worst < 0.05:
        print("=> FUSED KERNEL CORRECT vs torch SDPA on all calls.")
    else:
        dec = [r for r in rows if r[4] > 0.05 and not r[2]]
        pre = [r for r in rows if r[4] > 0.05 and r[2]]
        overflow_dec = [r for r in dec if r[5] > 100]
        print(f"=> KERNEL DIVERGES. decode-diverges={len(dec)} "
              f"prefill-diverges={len(pre)}")
        print(f"   decode-diverge w/ overflow(|T|>100)={len(overflow_dec)}")
        if overflow_dec:
            print("   => overflow IS present at decode diverge points; overflow "
                  "confirmed as cause (per SK.md, partial).")
        else:
            print("   => decode divergence has NO overflow; SK.md 'overflow "
                  "partial cause' is INSUFFICIENT for decode. Root cause is "
                  "elsewhere in the kernel (bf16 pack/chunk/softmax, not "
                  "overflow).")


if __name__ == "__main__":
    main()
