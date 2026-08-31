#!/usr/bin/env python3
"""Isolate whether the prefill DIVERGE is CAUSAL-MASK-related or overflow-related.

large_input_attn_test.py runs everything with causal=True. Here we run the
SAME overflow data but with causal=False (full attention, no future mask) to
see if the prefill divergence disappears. If causal=False -> OK, the bug is in
the causal mask wiring (not the LUT overflow floor). If still DIVERGE, it is
the LUT overflow.

Run: python examples/large_input_attn_nocausal.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
import numpy as np
import npu_gemm
from large_input_attn_test import make_overflow, npu_wrapper_ref


def run_case(name, n_rep, S_q, S_k, D=64, causal=False, sigma_q=80.0):
    if not npu_gemm.available():
        print(f"{name}: NPU NOT AVAILABLE -> skip\n"); return
    q, k, v, scale = make_overflow(n_rep, S_q, S_k, D, sigma_q=sigma_q)
    out_npu = npu_gemm.npu_attention_fused_impl(q, k, v, scale, n_rep, causal)
    out_host = npu_wrapper_ref(q, k, v, scale, n_rep, causal)
    diff = float(np.abs(out_npu - out_host).max())
    denom = float(np.abs(out_host).max()) + 1e-9
    cos = float(np.sum(out_npu * out_host) /
                ((np.linalg.norm(out_npu) * np.linalg.norm(out_host)) + 1e-30))
    nan = bool(np.isnan(out_npu).any() or np.isinf(out_npu).any())
    verdict = "DIVERGE" if (diff > 0.15 or nan) else "OK"
    print(f"{name} causal={causal} |host|max|={denom:7.2f} |npu|max|="
          f"{float(np.abs(out_npu).max()):7.2f}\n"
          f"  host-vs-npu dmax={diff:.4f} cos={cos:.6f} "
          f"{'NaN/Inf' if nan else ''} {verdict}\n")


def main():
    print("NPU:", npu_gemm.available())
    print("=== OVERFLOW with causal=FALSE (full attention) ===")
    run_case("overflow S_q=1  Sk=128 nocausal", 7, 1, 128)
    run_case("overflow S_q=41 Sk=64  nocausal", 7, 41, 64)
    run_case("overflow S_q=64 Sk=64  nocausal", 7, 64, 64)


if __name__ == "__main__":
    main()
