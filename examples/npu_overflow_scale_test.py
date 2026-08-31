#!/usr/bin/env python3
"""DECISIVE: is the real-input divergence due to OVERFLOW or a structural bug?

Strategy (SKAIL's real Q/K/V): capture real Q/K/V/scale/causal from one layer,
then scale Q by factor f. The kernel's score T = (Q*scale) @ K^T scales by f.
As f -> 0, max|T| -> 0 (well below bf16 exp range ~88) so overflow is removed.

  - If npu == host_ref when f is small  => OVERFLOW is the sole cause.
  - If npu STILL diverges when f is tiny => STRUCTURAL bug beyond overflow.

host_ref is the per-head softmax (proven == torch SDPA == wrapper unpacked).
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "server"))

os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"

import importlib, engine as engine_mod
importlib.reload(engine_mod)
import numpy as np
import torch
import torch.nn.functional as F
import npu_gemm


def host_ref(Q, K, V, scale, causal):
    """Independent per-head softmax on REAL packed Q/K/V. Returns f32."""
    n_hq, S_q, D = Q.shape
    Hkv = K.shape[0]; S_k = K.shape[1]; n_rep = n_hq // Hkv
    Qs = Q.astype(np.float32) * float(scale)
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(Hkv):
        q = Qs[g * n_rep:(g + 1) * n_rep]
        scores = q @ K[g].T
        if causal and S_q > 1:
            mask = np.triu(np.ones((S_q, S_k), bool), 1)
            scores = np.where(mask, -30.0, scores)
        z = scores - scores.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        out[g * n_rep:(g + 1) * n_rep] = P @ V[g]
    return out


def maxT_of(Q, K):
    mt = 0.0
    for g in range(K.shape[0]):
        mt = max(mt, float(np.abs(Q @ K[g].T).max()))
    return mt


def main():
    orig = npu_gemm.npu_attention_fused_impl
    calls = []

    def hook(Q, K, V, scale, n_rep=1, causal=False):
        out = orig(Q, K, V, scale, n_rep, causal)
        calls.append(dict(Q=Q.astype(np.float64), K=K.astype(np.float64),
                          V=V.astype(np.float64), scale=float(scale),
                          n_rep=n_rep, causal=bool(causal)))
        return out

    npu_gemm.npu_attention_fused_impl = hook

    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    print(f"engine: {len(e.layers)} layers\n")
    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    list(e.generate(PROMPT, {"max_tokens": 12, "temperature": 0.0, "top_k": 1}))
    print(f"captured {len(calls)} calls\n")

    # pick a mid decode call (S_q=1), e.g. call index 30
    pick = 30
    c = calls[pick]
    Q, K, V, scale, causal = c["Q"], c["K"], c["V"], c["scale"], c["causal"]
    print(f"=== decoding real call#{pick}: S_q={Q.shape[1]} "
          f"max|T|@f=1.0={maxT_of(Q, K):8.2f} ===\n")

    # validity: torch SDPA == host_ref at f=1 (proves host_ref is the truth)
    n_rep = c["n_rep"]
    n_hq = Q.shape[0]
    tq = torch.from_numpy(Q.astype(np.float32)); tk = torch.from_numpy(K.astype(np.float32))
    tv = torch.from_numpy(V.astype(np.float32))
    qg = tq * scale; kg = tk[None].expand(n_rep, -1, -1); vg = tv[None]
    torch_ref = F.scaled_dot_product_attention(qg[0:1], kg[0:1], vg[0:1]).numpy()[0]
    d_h = float(np.abs(host_ref(Q, K, V, scale, causal) - torch_ref).max())
    print(f"[validity] host_ref vs torch SDPA @ f=1: dmax={d_h:.6f} "
          f"({'OK' if d_h < 1e-4 else 'REF SUSPECT'})\n")

    print("=== scaling experiment (scale Q by 10^p) ===")
    for p in (0, -1, -2, -3, -4, -5, -6):
        f = 10.0 ** p
        Qs = Q * f
        m = maxT_of(Qs, K)
        host = host_ref(Qs, K, V, scale, causal)
        npu = npu_gemm.npu_attention_fused_impl(
            Qs.astype(np.dtype("bfloat16")), K, V, scale, n_rep, causal)
        dmax = float(np.abs(npu - host).max())
        denom = float(np.abs(host).max()) + 1e-9
        rd = dmax / denom
        cos = float(np.sum(npu * host) /
                    ((np.linalg.norm(npu) * np.linalg.norm(host)) + 1e-30))
        verdict = "OK" if rd < 1e-3 else "DIVERGE"
        print(f"  10^({p}) |f|={f:.1e} max|T|={m:10.4f} "
              f"rel_dmax={rd:.4f} cos={cos:.6f} -> {verdict}")


if __name__ == "__main__":
    main()
