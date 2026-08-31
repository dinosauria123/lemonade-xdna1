#!/usr/bin/env python3
"""NPU KERNEL overflow regression (reproduces SKAIL §1 large-input case).

Reproduces the overflow regime of the real model: Q scaled so that T=
(Q/sqrt(D)) @ K^T reaches ~2000 (bf16 exp range ~88 -> exp overflows).  Runs
the REAL NPU fused kernel (npu_attention_fused_impl) and compares to the host
wrapper reference (npu_wrapper_ref), which has been proven to match torch SDPA
exactly.  Reports cos / dmax / NaN flag.  This is empirical: no static LUT
theory (that hypothesis was refuted by the isolated unit test -- do NOT
re-assert).

Reproduce the "overflow" Q/K/V the real model hands the kernel:
  max|T| = |Q/sqrt(D) @ K^T| reaches ~2000 at L0; bf16 exp range ~88.
"""
import os, sys, math
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
import numpy as np
import npu_gemm


def npu_wrapper_ref(Q, K, V, scale, n_rep, causal):
    n_hq, S_q, D = Q.shape
    n_hkv = K.shape[0]; S_k = K.shape[1]
    Sk_pad = ((S_k + 63) // 64) * 64
    total = n_rep * S_q; Sq = ((total + 63) // 64) * 64
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        rs = np.arange(Sq); real = rs < total
        heads = np.where(real, rs // S_q, 0)
        poss = np.where(real, rs % S_q, 0)
        Qs = np.zeros((Sq, D), np.float32)
        Qs[real] = Q[g * n_rep + heads[real], poss[real]] * float(scale)
        Kt = np.zeros((D, Sk_pad), np.float32); Kt[:, :S_k] = K[g].T
        Vp = np.zeros((Sk_pad, D), np.float32); Vp[:S_k] = V[g]
        j = np.arange(Sk_pad)[None, :]
        pos_ok = (j <= poss[:, None]) if (causal and S_q > 1) else np.ones((Sq, Sk_pad), bool)
        allowed = real[:, None] & (j < S_k) & pos_ok
        M = np.where(allowed, 0.0, -30.0)
        T = Qs @ Kt + M
        z = T - T.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        out_unpack = P[:total] @ Vp[:S_k, :]
        for i in range(total):
            if real[i]:
                out[g * n_rep + heads[i], poss[i]] = out_unpack[i]
    return out


def make_overflow(n_rep, S_q, S_k, D=64, sigma_q=80.0, seed=1):
    """Construct Q/K/V bf16 (host-scaled). sigma_q large -> reproduces real L0
    overflow (|T|~2000); sigma_q=1.0 -> normal-range (|T|~1) kernel sanity."""
    rng = np.random.default_rng(seed)
    n_hq = n_rep  # single kv group for this test
    n_hkv = 1
    q = rng.standard_normal((n_hq, S_q, D)).astype(np.float32) * sigma_q
    k = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32) * sigma_q * 0.1
    v = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)
    scale = D ** -0.5
    return q.astype("bfloat16"), k.astype("bfloat16"), v.astype("bfloat16"), scale


def run_case(name, n_rep, S_q, S_k, D=64, causal=True, sigma_q=80.0, note=""):
    if not npu_gemm.available():
        print(f"{name}: NPU NOT AVAILABLE -> skip\n")
        return None
    q, k, v, scale = make_overflow(n_rep, S_q, S_k, D, sigma_q=sigma_q)
    # Report the real max|T| the kernel will see.
    Sk_pad = ((S_k + 63) // 64) * 64
    Qs = q.astype(np.float32) * scale
    T = np.einsum('rqd,id->rqi', Qs, k[0][:S_k])
    maxT = float(np.abs(T).max())
    out_npu = npu_gemm.npu_attention_fused_impl(q, k, v, scale, n_rep, causal)
    out_host = npu_wrapper_ref(q, k, v, scale, n_rep, causal)
    diff = float(np.abs(out_npu - out_host).max())
    denom = float(np.abs(out_host).max()) + 1e-9
    cos = float(np.sum(out_npu * out_host) / (
        (np.linalg.norm(out_npu) * np.linalg.norm(out_host)) + 1e-30))
    nan = bool(np.isnan(out_npu).any() or np.isinf(out_npu).any())
    verdict = "DIVERGE" if (diff > 0.15 or nan) else "OK"
    print(f"{name} | max|T|={maxT:8.2f} |Q|={float(np.abs(q).max()):5.1f} "
          f"|host|max|={denom:7.2f} |npu|max|={float(np.abs(out_npu).max()):7.2f}\n"
          f"  host-vs-npu dmax={diff:.4f} cos={cos:.6f} "
          f"{'NaN/Inf' if nan else ''} {verdict}\n")
    return diff, cos, nan


def main():
    print("NPU availability:", npu_gemm.available(),
          "| error:", npu_gemm.error()[:60])
    print("=== small (normal-range, should pass) ===")
    run_case("normal  S_q=1 Sk=128", 7, 1, 128, sigma_q=1.0)
    run_case("normal  S_q=41 Sk=64 causal", 7, 41, 64, sigma_q=1.0)
    run_case("CHUNKED normal S_q=64 Sk=64", 7, 64, 64, sigma_q=1.0,
             note="multi-chunk, |T| small")
    print("=== OVERFLOW (large |T|, reproduces real L0) ===")
    run_case("overflow S_q=1 Sk=128", 7, 1, 128)
    run_case("overflow S_q=41 Sk=64 causal", 7, 41, 64)
    run_case("overflow S_q=64 Sk=64 causal", 7, 64, 64)


if __name__ == "__main__":
    main()
