#!/usr/bin/env python3
"""Decisive: is the S_q=63 DIVERGE overflow(LUT)-driven or row-count-driven?

If sigma_q small (|T|~1, no LUT overflow) STILL diverges at S_q=63 -> it is a
row-count / packing bug, NOT the LUT. If small-sigma S_q=63 is OK but
large-sigma diverges -> it is the LUT overflow (and my -88 floor is not
taking effect, pointing at a JIT-cache / stale-kernel problem).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
import numpy as np
import npu_gemm
from large_input_attn_test import make_overflow, npu_wrapper_ref


def run(name, n_rep, S_q, S_k=64, sigma_q=80.0, causal=False):
    if not npu_gemm.available():
        print(f"{name}: NPU NA\n"); return
    q, k, v, scale = make_overflow(n_rep, S_q, S_k, 64, sigma_q=sigma_q)
    Sk_pad = ((S_k + 63) // 64) * 64
    Qs = q.astype(np.float32) * scale
    T = np.einsum('rqd,id->rqi', Qs, k[0][:S_k])
    maxT = float(np.abs(T).max())
    out_npu = npu_gemm.npu_attention_fused_impl(q, k, v, scale, n_rep, causal)
    out_host = npu_wrapper_ref(q, k, v, scale, n_rep, causal)
    diff = float(np.abs(out_npu - out_host).max())
    cos = float(np.sum(out_npu * out_host) /
                ((np.linalg.norm(out_npu) * np.linalg.norm(out_host)) + 1e-30))
    verdict = "DIVERGE" if (diff > 0.15) else "OK"
    print(f"{name:34s} max|T|={maxT:7.1f} dmax={diff:7.4f} cos={cos:.6f} {verdict}")


def main():
    print("NPU:", npu_gemm.available())
    print("--- small |T| (sigma_q=1, no LUT overflow expected) ---")
    run("n_rep=1 S_q=41 sigma=1", 1, 41, sigma_q=1.0)
    run("n_rep=1 S_q=63 sigma=1", 1, 63, sigma_q=1.0)
    run("n_rep=1 S_q=65 sigma=1", 1, 65, sigma_q=1.0)
    print("--- large |T| (sigma_q=80, overflow regime) ---")
    run("n_rep=1 S_q=41 sigma=80", 1, 41, sigma_q=80.0)
    run("n_rep=1 S_q=63 sigma=80", 1, 63, sigma_q=80.0)


if __name__ == "__main__":
    main()
