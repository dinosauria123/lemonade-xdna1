#!/usr/bin/env python3
"""Bisect the prefill DIVERGE by S_q (no causal mask, overflow data).

Decide: does it break at S_q>1, or only at the chunk boundary (Sq>320)?
  - n_rep=1, vary S_q: 1,2,4,8,16,41,63  -> total=S_q, Sq=ceil64(S_q)
    * S_q=1   -> 1 chunk (Sq=64)
    * S_q=41  -> 1 chunk (Sq=64)
    * S_q=63  -> 1 chunk (Sq=64)
    * S_q=64  -> 1 chunk (Sq=64)
    * S_q=65  -> 2 chunks (Sq=128)  <-- boundary
  - Also n_rep=7, S_q=41 -> total=287 Sq=320 (current failing case, 1 chunk)
  - n_rep=7, S_q=64 -> total=448 Sq=448 (2 chunks, failing)

causal=False so we isolate chunk/packing from the mask.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
import numpy as np
import npu_gemm
from large_input_attn_test import make_overflow, npu_wrapper_ref


def run(name, n_rep, S_q, S_k=64, D=64, causal=False, sigma_q=80.0):
    if not npu_gemm.available():
        print(f"{name}: NPU NA\n"); return
    q, k, v, scale = make_overflow(n_rep, S_q, S_k, D, sigma_q=sigma_q)
    out_npu = npu_gemm.npu_attention_fused_impl(q, k, v, scale, n_rep, causal)
    out_host = npu_wrapper_ref(q, k, v, scale, n_rep, causal)
    diff = float(np.abs(out_npu - out_host).max())
    cos = float(np.sum(out_npu * out_host) /
                ((np.linalg.norm(out_npu) * np.linalg.norm(out_host)) + 1e-30))
    nan = bool(np.isnan(out_npu).any() or np.isinf(out_npu).any())
    n_hq = n_rep
    total = n_rep * S_q; Sq = ((total + 63) // 64) * 64
    nchunk = (Sq + 319) // 320
    verdict = "DIVERGE" if (diff > 0.15 or nan) else "OK"
    print(f"{name:28s} Sq={Sq:4d} nchunk={nchunk} dmax={diff:7.4f} "
          f"cos={cos:.6f} {verdict}")


def main():
    print("NPU:", npu_gemm.available(), " (causal=False, overflow)")
    for S_q in [1, 2, 4, 8, 16, 41, 63, 65]:
        run(f"n_rep=1 S_q={S_q}", 1, S_q)
    run("n_rep=7 S_q=41", 7, 41)
    run("n_rep=7 S_q=64", 7, 64)


if __name__ == "__main__":
    main()
