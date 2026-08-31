#!/usr/bin/env python3
"""DECISIVE test: does the NPU fused-attention kernel match torch SDPA on the
SAME real Q/K/V?

The model computes attention with torch.nn.functional.scaled_dot_product_attention.
When NPU is enabled, engine routes through npu_attention_fused_impl. So: hook
that impl, capture each layer's REAL Q/K/V AND the NPU output it returned, and
compare to torch SDPA run on those exact tensors (= the model's true output).

    NPU == torch SDPA  (within bf16 tolerance)  => fused kernel is CORRECT -> revert NOT needed
    NPU != torch SDPA (beyond tolerance)        => fused kernel DIVERGES  -> revert IS needed

Unlike debug_fused_attn_3way / debug_fused_attn_decisive (which build independent
numpy references that diverge from torch SDPA themselves), this uses torch SDPA on
the unpacked MHA as ground truth and compares it directly to the recorded NPU
output. This is the SKILL.md-recommended bisection: real tensors, real NPU, and
the true model operator.
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "server"))

import numpy as np
import torch
import torch.nn.functional as F

XDNA_NPU_ATTENTION = "1"
os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"

import importlib, engine as engine_mod
importlib.reload(engine_mod)
import npu_gemm


def torch_ref_from_packed(raw_Q, raw_K, raw_V, scale, n_rep, causal, S_q):
    """torch SDPA on the unpacked GQA MHA = the model's true attention output.

    raw_Q: (n_hq, S_q, D) bf16-as-float, packed: row of Qs is group*g*n_rep+h.
    raw_K/raw_V: (n_hkv, S_k, D). n_hq = n_rep * n_hkv.
    Returns (n_hq, S_q, D) float32 — same layout as the NPU output.
    """
    n_hkv = raw_K.shape[0]
    S_k = raw_K.shape[1]
    D = raw_Q.shape[-1]
    scale = float(scale) if scale else D ** -0.5
    q = torch.from_numpy(raw_Q.astype(np.float32))    # (n_hq, S_q, D)
    k = torch.from_numpy(raw_K.astype(np.float32))    # (n_hkv, S_k, D)
    v = torch.from_numpy(raw_V.astype(np.float32))
    outs = []
    for g in range(n_hkv):
        qg = q[g * n_rep:(g + 1) * n_rep] * scale     # (n_rep, S_q, D)
        kg = k[g].unsqueeze(0).expand(n_rep, S_k, D)
        vg = v[g].unsqueeze(0).expand(n_rep, S_k, D)
        mask = None
        if causal and S_q > 1:
            mask = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool), 1)
        o = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=mask)
        outs.append(o)                                # (n_rep, S_q, D)
    return torch.cat(outs, dim=0).numpy()             # (n_hq, S_q, D)


def main():
    orig = npu_gemm.npu_attention_fused_impl
    calls = []
    def hook(Q, K, V, scale, n_rep=1, causal=False):
        out = orig(Q, K, V, scale, n_rep, causal)
        calls.append(dict(
            Q=Q.astype(np.float64), K=K.astype(np.float64), V=V.astype(np.float64),
            scale=float(scale), n_rep=n_rep, causal=bool(causal),
            n_hq=Q.shape[0], n_hkv=K.shape[0], npu=np.array(out, np.float64)))
        return out
    npu_gemm.npu_attention_fused_impl = hook

    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    print(f"built engine: {len(e.layers)} layers, npu_available={e.npu_available}\n")
    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    toks = list(e.generate(PROMPT, {"max_tokens": 6, "temperature": 0.0, "top_k": 1}))
    print("generated:", "".join(toks))
    print(f"captured {len(calls)} fused-attention calls\n")

    tol_rel = 0.10   # bf16 + LUT softmax tolerance on |output|
    worst = 0.0
    worst_call = None
    for i, c in enumerate(calls):
        Q, K, V = c["Q"], c["K"], c["V"]
        scale, n_rep, causal = c["scale"], c["n_rep"], c["causal"]
        S_q = Q.shape[1]
        ref = torch_ref_from_packed(Q, K, V, scale, n_rep, causal, S_q)
        npu = c["npu"]
        if ref.shape != npu.shape:
            print(f"  call#{i} SHAPE MISMATCH ref={ref.shape} npu={npu.shape} "
                  f"n_rep={n_rep} causal={int(causal)}")
            continue
        d = float(np.abs(ref - npu).max())
        denom = float(np.abs(ref).max()) + 1e-6
        rd = d / denom
        if rd > worst:
            worst = rd
            worst_call = i
        print(f"  call#{i:2d} n_rep={n_rep} causal={int(causal)} "
              f"S_q={S_q} n_hq={c['n_hq']} |ref|max|={denom:.2f} "
              f"dmax={d:8.4f} rel={rd:.4f} |npu|max|={float(np.abs(npu).max()):8.2f}")

    print(f"\n--- WORST RELATIVE DMAX = {worst:.4f} (call#{worst_call}) ---")
    if worst < tol_rel:
        print("VERDICT: NPU == torch SDPA within bf16 tolerance -> "
              "fused kernel is CORRECT (revert NOT needed).")
    else:
        print("VERDICT: NPU != torch SDPA beyond tolerance -> "
              "fused kernel DIVERGES (revert IS needed).")


if __name__ == "__main__":
    main()
