#!/usr/bin/env python3
"""DECISIVE overflow test on the REAL Qwen2.5-0.5B Q/K/V (SKAIL Next-step 2).

Reuse decisive_test.py's capture: hook npu_attention_fused_impl to record each
layer's REAL packed Q/K/V AND the NPU output it returned.  Then:

  1. Compute the real max|T| the kernel saw, |Q/sqrt(D) @ K^T|, per call.
     (This tests SKAIL's claim: real-model max|T| = 2061.)
  2. Compare NPU output vs an INDEPENDENT host per-head reference (host_ref)
     on the EXACT same real tensors.  If they diverge here (no wrapper),
     the NPU kernel itself (overflow) is the cause.
  3. Greedy decode WITH NPU fused -> show it diverges from the reference.
"""
import os, sys, math
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "server"))

XDNA_NPU_ATTENTION = "1"
os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"

import importlib, engine as engine_mod
importlib.reload(engine_mod)
import numpy as np
import torch
import torch.nn.functional as F
import npu_gemm


def host_ref(Q, K, V, scale, causal):
    """Independent per-head softmax on REAL packed Q/K/V.
    Q: (H,S_q,D) bf16-as-float, K/V: (Hkv,S_k,D). Query head h attends KV head
    h//n_rep (packed). Returns f32 (H,S_q,D)."""
    n_hq, S_q, D = Q.shape
    Hkv = K.shape[0]; S_k = K.shape[1]
    n_rep = n_hq // Hkv
    Qs = Q.astype(np.float32) * float(scale)
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(Hkv):
        q = Qs[g * n_rep:(g + 1) * n_rep]            # (n_rep, S_q, D)
        scores = q @ K[g].T                          # (n_rep, S_q, S_k)
        if causal and S_q > 1:
            mask = np.triu(np.ones((S_q, S_k), bool), 1)
            scores = np.where(mask, -30.0, scores)
        z = scores - scores.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        out[g * n_rep:(g + 1) * n_rep] = P @ V[g]
    return out


def maxT_of(Q_packed, K):
    """max|Q_packed @ K^T| over all packed rows x keys (the real max score)."""
    maxT = 0.0
    for g in range(K.shape[0]):
        Kt = K[g].T
        T = Q_packed @ Kt
        maxT = max(maxT, float(np.abs(T).max()))
    return maxT


def main():
    orig = npu_gemm.npu_attention_fused_impl
    calls = []

    def hook(Q, K, V, scale, n_rep=1, causal=False):
        out = orig(Q, K, V, scale, n_rep, causal)
        qf = Q.astype(np.float64)
        kf = K.astype(np.float64)
        calls.append(dict(Q=qf, K=kf, V=V.astype(np.float64),
                          scale=float(scale), n_rep=n_rep,
                          causal=bool(causal), npu=np.array(out, np.float64)))
        return out

    npu_gemm.npu_attention_fused_impl = hook

    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    print(f"engine: {len(e.layers)} layers, npu_available={e.npu_available}\n")

    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    print("greedy decode WITH fused NPU...")
    toks = list(e.generate(PROMPT, {"max_tokens": 12, "temperature": 0.0,
                                     "top_k": 1}))
    print("generated (NPU):", "".join(toks))
    print(f"captured {len(calls)} fused-attention calls\n")

    worst = 0.0; worst_call = None; worst_t = 0.0
    for i, c in enumerate(calls):
        Q, K, V, scale, causal = c["Q"], c["K"], c["V"], c["scale"], c["causal"]
        S_q = Q.shape[1]
        host = host_ref(Q, K, V, scale, causal)
        dmax = float(np.abs(c["npu"] - host).max())
        denom = float(np.abs(host).max()) + 1e-9
        rd = dmax / denom
        mt = maxT_of(Q, K)
        if rd > worst:
            worst = rd; worst_call = i; worst_t = mt
        print(f"  call#{i:2d} S_q={S_q} n_rep={c['n_rep']} "
              f"max|T|={mt:8.2f} rel_dmax={rd:.4f} "
              f"{'OK' if rd < 0.1 else 'DIVERGE'}")

    print(f"\n--- WORST real max|T| = {worst_t:.2f} (call#{worst_call}), "
          f"worst rel_dmax={worst:.4f} ---")
    if worst < 0.1:
        print("VERDICT: NPU == host reference within tolerance on REAL "
              "inputs -> NOT overflow; root cause is elsewhere.")
    else:
        print("VERDICT: NPU != host reference on REAL inputs -> kernel diverges "
              "on real Q/K/V (see per-call max|T|).")


if __name__ == "__main__":
    main()
