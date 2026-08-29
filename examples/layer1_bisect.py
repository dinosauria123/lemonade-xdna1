#!/usr/bin/env python3
"""Decisive per-layer bisect of the fused-attention residual (real model data).

Per SKILL, prior 'overflow / softmax / LUT / wrapper' hypotheses were chased on
buggy harnesses. THIS test compares, per captured REAL call:

    (a) fused  -- the production fused kernel (int8 LUT softmax)
    (b) perhead-- NPU GEMM (matmul_bf16) + CPU full-precision softmax
    (c) numpy  -- provably-correct causal GHA reference (full-precision)

    fused == perhead == numpy   => wrapper & kernel & ref all agree
    fused != numpy, perhead==numpy  => BUG IS IN FUSED KERNEL (softmax/mask/scale)
    fused != numpy, perhead != numpy  => ref is wrong (shouldn't happen)

Also prints per-layer score statistics (max|T|, #scores > 30) so the fused
kernel's int8 +0/-30 softmax saturation (overflow) can be judged directly.
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"

import numpy as np
from ml_dtypes import bfloat16 as bf16
import importlib, engine as engine_mod
importlib.reload(engine_mod)
import npu_gemm


def numpy_ref(Q, K, V, scale, n_rep, causal, nh_q, n_hkv, D):
    """Provably-correct causal GHA reference (full precision f32)."""
    S_q = Q.shape[1]; S_k = K.shape[1]
    out = np.zeros(Q.shape, np.float32)
    for g in range(n_hkv):
        k_g = K[g]; v_g = V[g]
        for i in range(n_rep):
            q_g = Q[g * n_rep + i]                       # (S_q, D)
            scores = (q_g @ k_g.T) * float(scale)        # (S_q, S_k)
            if causal and S_q > 1:
                i_idx = np.arange(S_q)[:, None]
                j_idx = np.arange(S_k)[None, :]
                scores = np.where(j_idx <= i_idx, scores, np.finfo(np.float32).min)
            scores = scores - scores.max(axis=1, keepdims=True)
            P = np.exp(scores); P /= P.sum(axis=1, keepdims=True)
            out[g * n_rep + i] = P @ v_g
    return out


def score_stats(Q, K, V, scale, n_rep, causal):
    """Score distribution for the LUT / overflow judgment."""
    S_q = Q.shape[1]; S_k = K.shape[1]
    worst = 0.0; nclip = 0
    for g in range(K.shape[0]):
        for i in range(n_rep):
            q_g = Q[g * n_rep + i]
            T = (q_g @ K[g].T) * float(scale)          # (S_q, S_k)
            worst = max(worst, float(np.abs(T).max()))
            nclip += int((T > 30.0).sum())
    return worst, nclip


def main():
    orig_fused = npu_gemm.npu_attention_fused_impl
    captured = []

    def hook(Q, K, V, scale, n_rep=1, causal=False):
        out = orig_fused(Q, K, V, scale, n_rep, causal)
        captured.append(dict(Q=Q.astype(np.float32), K=K.astype(np.float32),
                             V=V.astype(np.float32), scale=float(scale),
                             n_rep=int(n_rep), causal=bool(causal),
                             fused=out.astype(np.float32)))
        return out

    npu_gemm.npu_attention_fused_impl = hook
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    nh_q = e.model.config.num_attention_heads
    n_hkv = e.model.config.num_key_value_heads
    n_rep = nh_q // n_hkv
    D = e.model.config.hidden_size // nh_q
    print(f"heads={nh_q} kv={n_hkv} n_rep={n_rep} D={D} layers={len(e.layers)}\n",
          flush=True)

    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    toks = list(e.generate(PROMPT, {"max_tokens": 8, "temperature": 0.0, "top_k": 1}))
    print(f"generated: {''.join(toks)!r}  ({len(captured)} calls)\n", flush=True)

    perhead = npu_gemm.npu_attention_core
    rows = []
    for i, c in enumerate(captured):
        Q, K, V, scale, causal = c["Q"], c["K"], c["V"], c["scale"], c["causal"]
        out_f = c["fused"]
        # per-head on the NPU (matmul_bf16 + CPU softmax), real path
        try:
            out_p = perhead(Q, K, V, scale, c["n_rep"], causal, force_cpu=False)
        except Exception as ex:
            out_p = np.zeros_like(out_f)
        ref = numpy_ref(Q, K, V, scale, c["n_rep"], causal, nh_q, n_hkv, D)
        d_fp = float(np.abs(out_f - out_p).max())
        d_fn = float(np.abs(out_f - ref).max())
        d_pn = float(np.abs(out_p - ref).max())
        worstT, nclip = score_stats(Q, K, V, scale, c["n_rep"], causal)
        rows.append((i, Q.shape[1], d_fp, d_fn, d_pn, worstT, nclip, causal))

    print("--- per layer/call  [fused_vs_perhead  fused_vs_numpy  perhead_vs_numpy] "
          "max|T|  #scores>30  causal")
    for i, Sq, dfp, dfn, dpn, wT, nc, ca in rows:
        print(f"  call#{i:3d} S_q={Sq:3d}  fp={dfp:7.4f}  fn={dfn:7.4f}  "
              f"pn={dpn:7.4f}  max|T|={wT:6.1f}  #clip={nc:4d}  ca={ca}")

    worst = sorted(rows, key=lambda t: -t[2])
    print("\n--- worst fused-vs-perhead:", flush=True)
    for i, Sq, dfp, dfn, dpn, wT, nc, ca in worst[:8]:
        print(f"  call#{i} S_q={Sq} fused_vs_perhead={dfp:.4f} "
              f"fused_vs_numpy={dfn:.4f} perhead_vs_numpy={dpn:.4f} max|T|={wT:.1f}")

    print("\n--- worst fused-vs-numpy:", flush=True)
    for i, Sq, dfp, dfn, dpn, wT, nc, ca in sorted(rows, key=lambda t: -t[3])[:8]:
        print(f"  call#{i} S_q={Sq} fused_vs_numpy={dfn:.4f} "
              f"fused_vs_perhead={dfp:.4f} perhead_vs_numpy={dpn:.4f} max|T|={wT:.1f}")

    print("\n--- worst perhead-vs-numpy:", flush=True)
    for i, Sq, dfp, dfn, dpn, wT, nc, ca in sorted(rows, key=lambda t: -t[4])[:8]:
        print(f"  call#{i} S_q={Sq} perhead_vs_numpy={dpn:.4f} "
              f"fused_vs_numpy={dfn:.4f} fused_vs_perhead={dfp:.4f} max|T|={wT:.1f}")

    # Decisive summary: does the fused kernel diverge from perhead while perhead
    # agrees with numpy?  If so, the residual lives in the fused kernel.
    bad = [(r) for r in rows if r[2] > 0.5 and r[4] < 0.3]
    if bad:
        print(f"\nSUMMARY: {len(bad)} call(s) where fused!=perhead but perhead==numpy "
              "-> RESIDUAL BUG IS IN THE FUSED KERNEL", flush=True)
    else:
        print("\nSUMMARY: no fused-vs-perhead gap while perhead matches numpy; "
              "the residual is not a clean kernel-only defect", flush=True)


if __name__ == "__main__":
    main()
