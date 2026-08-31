#!/usr/bin/env python3
"""Three-way disambiguation on ONE layer's REAL tensors (SKILL Next step 2).

Feeds one layer's real Q/K/V into three paths and reports pairwise divergence:

  npu_fused -- the production NPU fused path (npu_attention_fused_impl)
  cpu_ref   -- the engine's OWN cpu reference (npu_attention_core, force_cpu)
               This is the path the model was validated against at build time.
  numpy     -- an independent packed softmax reference (my own)

  npu_fused == cpu_ref   ==> model is fine; any prior NPU "divergence" vs MY
                             numpy ref means the BUG was in my reference, not the
                             model or NPU (most likely when only *I* diverge).
  npu_fused != cpu_ref   ==> REAL kernel vs engine reference (kernel/packing bug).
  numpy != cpu_ref        ==> my numpy ref is malformed (packing/mask error).

Usage: examples/debug_fused_attn_3way.py [LAYER]   (default: 21)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"

import numpy as np
from ml_dtypes import bfloat16 as bf16
import importlib, engine as engine_mod; importlib.reload(engine_mod)
import npu_gemm


def _ref_attention(Qs, Kt, V, M):
    T = Qs @ Kt + M
    z = T - T.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)) @ V


def _numpy_case(Q, K, V, scale, n_rep, causal):
    """Independent packed softmax reference, mirroring the real model layout."""
    n_hq, S_q, D = Q.shape
    n_hkv = K.shape[0]
    S_k = K.shape[1]
    if n_rep == 1 and n_hkv != n_hq:
        n_rep = n_hq // n_hkv
    Sk_pad = ((S_k + 63) // 64) * 64
    total = n_rep * S_q
    Sq = ((total + 63) // 64) * 64
    rs = np.arange(Sq); real = rs < total
    heads = np.where(real, rs // S_q, 0)
    poss = np.where(real, rs % S_q, 0)
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        Qsg = np.zeros((Sq, D), np.float32)
        Qsg[real] = Q[g * n_rep + heads[real], poss[real]].astype(np.float32) * float(scale)
        Kt = np.zeros((D, Sk_pad), np.float32); Kt[:, :S_k] = K[g].T
        Vp = np.zeros((Sk_pad, D), np.float32); Vp[:S_k] = V[g]
        j = np.arange(Sk_pad)[None, :]
        pos_ok = ((j <= poss[:, None]) if (causal and S_q > 1)
                  else np.ones((Sq, Sk_pad), bool))
        allowed = real[:, None] & (j < S_k) & pos_ok
        M = np.where(allowed, 0.0, -30.0)
        P = _ref_attention(
            Qsg.astype(bf16).astype(np.float32),
            Kt.astype(bf16).astype(np.float32),
            Vp.astype(bf16).astype(np.float32),
            M.astype(bf16).astype(np.float32))
        out[g * n_rep:(g + 1) * n_rep] = P[:S_q]
    return out


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    data = []
    npu_out = {}
    orig = npu_gemm.npu_attention_fused_impl

    def hook(Q, K, V, scale, n_rep=1, causal=False):
        data.append(dict(Q=Q.copy(), K=K.copy(), V=V.copy(),
                         scale=float(scale), n_rep=n_rep, causal=causal))
        return orig(Q, K, V, scale, n_rep, causal)

    npu_gemm.npu_attention_fused_impl = hook
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    list(e.generate([{"role": "user", "content": "hi"}],
                    {"max_tokens": 6, "temperature": 0.0, "top_k": 1}))

    print(f"recorded {len(data)} layers\n", flush=True)
    if layer >= len(data):
        print(f"layer {layer} out of range (0..{len(data)-1}); using {layer % len(data)}")
        layer %= len(data)

    d = data[layer]
    Q, K, V = d["Q"], d["K"], d["V"]
    scale, n_rep, causal = d["scale"], d["n_rep"], d["causal"]
    n_hq, S_q, D = Q.shape
    n_hkv = K.shape[0]
    print(f"layer {layer}: n_hq={n_hq} S_q={S_q} D={D} S_k={K.shape[1]} "
          f"n_rep={n_rep} causal={causal} scale={scale:.5f}\n", flush=True)

    # (a) production NPU fused path output (already captured during generate)
    fused = orig(Q, K, V, scale, n_rep, causal)

    # (b) engine's own cpu reference for the SAME tensors
    cpu = npu_gemm.npu_attention_core(Q, K, V, scale, n_rep, causal, force_cpu=True)

    # (c) independent numpy reference
    ref = _numpy_case(Q, K, V, scale, n_rep, causal)

    dNPU_cpu = float(np.abs(fused.astype(np.float64) - cpu.astype(np.float64)).max())
    dNPU_np  = float(np.abs(fused.astype(np.float64) - ref.astype(np.float64)).max())
    d_cpu_np = float(np.abs(cpu.astype(np.float64) - ref.astype(np.float64)).max())

    print("=== THREE-WAY pairwise dmax (abs) ===")
    print(f"  npu_fused vs engine_cpu : {dNPU_cpu:.6g}")
    print(f"  npu_fused vs numpy_ref  : {dNPU_np:.6g}")
    print(f"  engine_cpu  vs numpy_ref: {d_cpu_np:.6g}\n", flush=True)

    print(f"  |Q|max={np.abs(Q).max():.2f} |K|max={np.abs(K).max():.2f} "
          f"|V|max={np.abs(V).max():.2f}", flush=True)
    print(f"  L2(fused)={np.linalg.norm(fused):.4f} "
          f"L2(cpu)={np.linalg.norm(cpu):.4f} "
          f"L2(ref)={np.linalg.norm(ref):.4f}", flush=True)

    if np.linalg.norm(ref) < 1e-3:
        print("\n*** numpy ref is near-zero (malformed) -> my reference is the bug",
              flush=True)
    if dNPU_cpu < 1e-2 and d_cpu_np < 1e-2:
        print("\n*** npu_fused == engine_cpu == numpy_ref : model is FINE.",
              "any earlier E2E complaint was an artifact of a different path.",
              flush=True)
    elif dNPU_cpu < 1e-2 and d_cpu_np > 1e-2:
        print("\n*** npu_fused == engine_cpu, but numpy_ref differs => MY numpy ref "
              "is malformed (pack/mask bug), not the model.", flush=True)
    elif dNPU_cpu > 1e-2:
        print("\n*** npu_fused != engine_cpu : REAL kernel-vs-reference divergence.",
              flush=True)


if __name__ == "__main__":
    main()
