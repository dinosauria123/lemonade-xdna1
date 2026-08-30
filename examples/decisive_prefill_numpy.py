#!/usr/bin/env python3
"""DECISIVE DIFFERENT ROUTE #2 -- pure-numpy, no torch, no model download.

Question left after SKILL.md: is the prefill-only residual a HOST-MATH bug
(packing / causal-mask / AV-unpack in npu_attention_fused_impl) or is it NPU
bf16 GEMM quantization (fixable only by disabling fused / better kernel)?

All prior forks compared implementations that SHARE the tight-packed layout ->
they cannot distinguish "host packing wrong" from "NPU quant wrong". This test
uses an INDEPENDENT ground truth (truly unpacked per-query-head MHA, identical
semantics to torch SDPA -- no shared packing) and compares:

    A  unpacked_ref  -- independent ground truth (per-query-head MHA)
    B  fused_host_ref -- tight-packed host math of the REAL impl (float32)
    C  npu fused_out  -- actual NPU bf16 GEMM (best effort)

VERDICT:
    A==B  AND  (C missing or B!=C) => residual is NPU bf16 GEMM QUANTIZATION;
                                     host packing/mask/AV is CORRECT.
    A!=B                            => HOST-MATH bug in tight-packed prefill path.
    A==B  AND  B==C                 => no residual (overflow benign on host).

Two regimes are tested to separate confounds:
    NORMAL (sigma=1)   : isolates packing/mask/AV correctness (no overflow).
    OVERFLOW (sigma=80): reproduces real L0 overflow (|T|~2000).
"""
import numpy as np
from ml_dtypes import bfloat16

D = 64; n_rep = 7; n_hkv = 2; n_hq = n_rep * n_hkv


def packed_host(Q, K, V, scale, causal, S_q, S_k):
    """Exact host replica of npu_attention_fused_impl tight-packed path (f32)."""
    total = n_rep * S_q
    Sq = ((total + 63) // 64) * 64
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        rs = np.arange(Sq); real = rs < total
        heads = np.where(real, rs // S_q, 0); poss = np.where(real, rs % S_q, 0)
        Qs = np.zeros((Sq, D)); Qs[real] = Q[g*n_rep+heads[real], poss[real]]*scale
        T = Qs @ K[g].T
        j = np.arange(S_k)[None, :]
        pos_ok = (j <= poss[:, None]) if causal else np.ones((Sq, S_k), bool)
        allowed = real[:, None] & (j < S_k) & pos_ok
        T = np.where(allowed, T, -np.inf)
        z = T - np.nanmax(T, axis=1, keepdims=True)
        P = np.exp(z); P = np.nan_to_num(P); P /= P.sum(axis=1, keepdims=True)
        O = P @ V[g]
        out[g*n_rep+heads[real], poss[real]] = O[real]
    return out


def unpacked_ref(Q, K, V, scale, causal, S_q, S_k):
    """Independent ground truth: per-query-head MHA (NO shared packing)."""
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        for h in range(n_rep):
            q = Q[g*n_rep+h].astype(np.float32)*scale
            T = q @ K[g].T
            i = np.arange(S_q)[:, None]; j = np.arange(S_k)[None, :]
            if causal: T = np.where(i >= j, T, -np.inf)
            T = T - T.max(axis=1, keepdims=True)
            P = np.exp(T); P /= P.sum(axis=1, keepdims=True)
            out[g*n_rep+h] = P @ V[g]
    return out


def mk(n_rep, S_q, S_k, sigma, seed=1):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_hq, S_q, D)).astype(np.float32)*sigma
    k = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)*sigma*0.1
    v = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)
    return q, k, v


def rel(A, B):
    d = float(np.abs(A-B).max()); return d, d/(float(np.abs(A).max())+1e-9)


def run(name, sigma, causal=True):
    scale = D**-0.5
    S_q, S_k = 41, 64
    q, k, v = mk(n_rep, S_q, S_k, sigma)
    A = unpacked_ref(q, k, v, scale, causal, S_q, S_k)
    B = packed_host(q, k, v, scale, causal, S_q, S_k)
    dB, rB = rel(A, B)
    C = None
    try:
        import os, sys
        sys.path.insert(0, "/home/dino/open-xdna/server")
        os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
        import npu_gemm
        if npu_gemm.available():
            C = npu_gemm.npu_attention_fused_impl(
                q.astype(bfloat16), k.astype(bfloat16), v.astype(bfloat16),
                scale, n_rep, causal)
    except Exception as ex:
        print(f"   NPU: {type(ex).__name__}: {ex}")
    rC = None
    if C is not None:
        _, rC = rel(A, C)
    maxT = float(np.abs(q*scale @ k[0].T).max())
    print(f"  {name} max|T|={maxT:8.2f} "
          f"A-vs-B rel={rB:.6f}  B-vs-C rel={'?' if rC is None else f'{rC:.6f}'} "
          f"{'(NPU)' if C is None else '(npu)'}")
    return rB, rC


def main():
    print("=== NORMAL (sigma=1): isolates packing/mask/AV ===")
    rnB, rnC = run("normal  S_q=41 Sk=64", 1.0)
    print("\n=== OVERFLOW (sigma=80): real L0 regime ===")
    roB, roC = run("overflow S_q=41 Sk=64", 80.0)
    print("\n=== VERDICT (別経路 / independent ground truth) ===")
    if rnB > 0.05 or roB > 0.05:
        print("  => HOST-MATH BUG in tight-packed prefill packing/mask/AV.\n"
              "     Fix npu_attention_fused_impl host path (host != unpacked "
              "ground truth).")
    elif rnC is None and roC is None:
        print("  => NPU unavailable this run; only host-vs-truth established "
              "(host CORRECT).")
    elif max(x for x in (rnC, roC) if x is not None) > 0.15:
        print("  => RESIDUAL IS NPU bf16 GEMM QUANTIZATION (host path CORRECT).\n"
              "     packing/mask/AV match the independent ground truth; "
              "divergence is bf16 quant, not overflow-floor.")
    else:
        print("  => no significant residual on this capture.")


if __name__ == "__main__":
    main()
