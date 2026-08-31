#!/usr/bin/env python3
"""NPU fused-attention overflow regression -- MINIMAL, decisive.

NO wrapper/packing math here. Two independent implementations compared
directly on the SAME real Q/K/V:

  host_ref() -- per-head numpy softmax (independent of the kernel code)
  npu_gemm.npu_attention_fused_impl() -- the real AIE2 fused kernel

Single KV head, one attention group, D=64 (Qwen2.5-0.5B). Two cases differ
in ONLY one variable, the input magnitude:
  - normal:  sigma=1.0  -> |T| = |Q/sqrt(D) @ K^T| ~ 0.4   (bf16 exp safe)
  - overflow: sigma=80 -> |T| ~ 2000                         (bf16 exp OVERFLOW)

If they agree on 'normal' and diverge on 'overflow', the NPU kernel is the
cause (overflow), not the wrapper.
"""
import os, sys, math
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
import numpy as np
import npu_gemm


def host_ref(Q, K, V, scale, causal):
    """Independent per-head softmax. Q: (H,S_q,D), K/V: (1,S_k,D). returns f32."""
    H, S_q, D = Q.shape
    S_k = K.shape[1]
    Qs = Q.astype(np.float32) * float(scale)      # pre-scale like the wrapper
    out = np.zeros((H, S_q, D), np.float32)
    for h in range(H):
        q = Qs[h]                                  # (S_q, S_k) after @ Kt
        scores = q @ K[0].T                        # (S_q, S_k)
        if causal and S_q > 1:
            mask = np.triu(np.ones((S_q, S_k), bool), 1)
            scores = np.where(mask, -30.0, scores)
        z = scores - scores.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        out[h] = P @ V[0]
    return out


def make(H, S_q, S_k, D=64, sigma=80.0, seed=1):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((H, S_q, D)).astype(np.float32) * sigma
    K = rng.standard_normal((1, S_k, D)).astype(np.float32) * sigma * 0.1
    V = rng.standard_normal((1, S_k, D)).astype(np.float32)
    scale = D ** -0.5
    return Q.astype(np.dtype("bfloat16")), K.astype(np.dtype("bfloat16")), \
        V.astype(np.dtype("bfloat16")), scale


def cos(a, b):
    return float(np.sum(a * b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-30))


def run(name, sigma):
    H, S_q, S_k = 1, 1, 128
    Q, K, V, scale = make(H, S_q, S_k, sigma=sigma)
    # |T| the kernel will see: Qs[h] @ K[0].T for h=0
    T = Q[0] @ K[0].T
    maxT = float(np.abs(T).max())
    ref = host_ref(Q, K, V, scale, causal=True)
    npu = npu_gemm.npu_attention_fused_impl(Q, K, V, scale, 1, True)
    dmax = float(np.abs(npu - ref).max())
    c = cos(npu, ref)
    nan = np.isnan(npu).any() or np.isinf(npu).any()
    verdict = "DIVERGE" if (nan or dmax > 0.15) else "OK"
    print(f"{name} (sigma={sigma}) |maxT|={maxT:9.2f} "
          f"|ref|max|={float(np.abs(ref).max()):6.2f} "
          f"|npu|max|={float(np.abs(npu).max()):6.2f} "
          f"dmax={dmax:.5f} cos={c:.6f} "
          f"{'NaN' if nan else ''} -> {verdict}")


def main():
    print("NPU:", npu_gemm.available(), "| err:", npu_gemm.error()[:50])
    run("normal   ", 1.0)
    run("overflow ", 80.0)


if __name__ == "__main__":
    main()
