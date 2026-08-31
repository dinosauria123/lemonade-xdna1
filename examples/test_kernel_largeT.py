#!/usr/bin/env python3
"""Run the EXISTING kernel unit-test (_run_case in mm_attention_llm.py) with
LARGE |T| (sigma=80) to see if the kernel itself diverges at large scores --
isolating kernel vs wrapper.  Small |T| is known ALL-PASS (cos 0.9999+).

Run: python examples/test_kernel_largeT.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
import numpy as np
# import the attention module to reuse _run_case
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "mlir-aie", "programming_examples", "ml", "attention"))
import mm_attention_llm as attn
import npu_gemm

# Monkeypatch the data generator inside _run_case to use a large sigma so
# |T| = |Qs @ Kt| reaches ~2000 like the real model.
_orig_run = attn._run_case

def patched_run_case(name, n_heads, S_q, S_k, causal, D=64, verbose=True,
                     sigma=80.0):
    # re-run the body but with scaled inputs: we can't easily inject sigma into
    # _run_case, so instead wrap: call original but the generator uses rng.normal
    # already; we instead patch numpy.default_rng? Simpler: call _ref + kernel
    # via the public fused_llm_attention with our own large-sigma data.
    return _orig_run_case_large(name, n_heads, S_q, S_k, causal, D, verbose, sigma)


def _orig_run_case_large(name, n_heads, S_q, S_k, causal, D, verbose, sigma):
    import math
    from ml_dtypes import bfloat16 as bf16
    bf = bf16
    rng = np.random.default_rng(1234)
    scale = 1.0 / math.sqrt(float(D))
    Qh = (rng.normal(0.0, 1.0, size=(n_heads, S_q, D)).astype(np.float32) * sigma).astype(np.float32)
    Kh = (rng.normal(0.0, 1.0, size=(S_k, D)).astype(np.float32) * sigma * 0.1).astype(np.float32)
    Vh = rng.normal(0.0, 1.0, size=(S_k, D)).astype(np.float32)
    Sk = ((S_k + 63) // 64) * 64
    total = n_heads * S_q
    Sq = ((total + 63) // 64) * 64
    rs = np.arange(Sq)
    real = rs < total
    heads = np.where(real, rs // S_q, 0)
    poss = np.where(real, rs % S_q, 0)
    Qs = np.zeros((Sq, D), dtype=np.float32)
    Qs[real] = Qh[heads[real], poss[real]] * scale
    Kt = np.zeros((D, Sk), dtype=np.float32); Kt[:, :S_k] = Kh.T
    Vp = np.zeros((Sk, D), dtype=np.float32); Vp[:S_k] = Vh
    j = np.arange(Sk)[None, :]
    posok = (j <= poss[:, None]) if causal else np.ones((Sq, Sk), dtype=bool)
    allowed = real[:, None] & (j < S_k) & posok
    M = np.where(allowed, 0.0, -30.0)
    M_i8 = np.where(allowed, 0, 1).astype(np.int8)
    import aie.iron as iron
    Kt_t = iron.tensor(Kt.reshape(-1).astype(bf), dtype=bf, device="npu")
    V_t = iron.tensor(Vp.reshape(-1).astype(bf), dtype=bf, device="npu")
    out = np.zeros((n_heads, S_q, D), dtype=np.float32)
    row0 = 0; ndispatch = 0
    while row0 < Sq:
        c = min(attn.MAX_BLOCKS_PER_DISPATCH * 64, Sq - row0)
        Qs_t = iron.tensor(Qs[row0:row0 + c].reshape(-1).astype(bf), dtype=bf, device="npu")
        M_t = iron.tensor(M_i8[row0:row0 + c].reshape(-1), dtype=np.int8, device="npu")
        O_t = iron.zeros(c * D, dtype=bf, device="npu")
        attn.fused_llm_attention(Qs_t, Kt_t, V_t, M_t, O_t, Sq=c, Sk=Sk, D=D, element_type=bf)
        O_np = O_t.numpy().copy().reshape(c, D)
        rsl = rs[row0:row0 + c]; reall = real[row0:row0 + c]
        out[heads[row0:row0 + c][reall], poss[row0:row0 + c][reall]] = O_np[rsl[reall] - row0]
        row0 += c; ndispatch += 1
    O_ref = attn._ref_attention(Qs.astype(bf).astype(np.float32),
                                Kt.astype(bf).astype(np.float32),
                                Vp.astype(bf).astype(np.float32),
                                M.astype(bf).astype(np.float32))
    ok = True
    for h in range(n_heads):
        act = out[h]; exp_ = O_ref[h * S_q:(h + 1) * S_q, :]
        cos = float((act * exp_).sum() / (np.linalg.norm(act) * np.linalg.norm(exp_) + 1e-30))
        dmax = float(np.abs(act - exp_).max())
        good = np.allclose(act, exp_, rtol=0.15, atol=0.02)
        ok = ok and good
        print(f"  head {h}: cos={cos:.6f} dmax={dmax:.5f} {'OK' if good else 'FAIL'}")
    print(f"{name}: {'PASS' if ok else 'FAIL'} (large |T|, Sq={Sq} Sk={Sk} ndispatch={ndispatch})")
    return ok


def main():
    if not npu_gemm.available():
        print("NPU not available"); return
    # large |T| versions of the unit-test shapes
    patched_run_case("gqa prefill 7x41 causal LARGE|T|", 7, 41, 41, True)
    patched_run_case("gqa prefill 7x64 causal LARGE|T|", 7, 64, 64, True)
    patched_run_case("gqa decode 7x1 Sk=128 LARGE|T|", 7, 1, 100, False)


if __name__ == "__main__":
    main()
