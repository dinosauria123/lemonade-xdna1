#!/usr/bin/env python3
"""Isolated bisection of the M3 fused-attention E2E divergence (real model data).

Per SKILL "next steps": dump each layer's REAL Q/K/V, route the SAME tensors
into (a) a numpy reference (_ref_attention, -30 mask) and (b) the real NPU
npu_attention_fused_impl, compare per layer.

    per-layer dmax ~ 0  ==> NPU agrees with numpy; divergence is at
                           wrapper/accumulator/other path
    per-layer dmax >> 0 ==> NPU kernel mishandles that layer's data
                           (bf16 range / LUT softmax)

Host packing mirrors the wrapper, so this isolates NPU-kernel-vs-ref on real
data.
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"

import numpy as np
from ml_dtypes import bfloat16 as bf16
import importlib, engine as engine_mod
importlib.reload(engine_mod)
import npu_gemm


def _ref_attention(Qs, Kt, V, M):
    T = Qs @ Kt + M
    z = T - T.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)) @ V


def _ref_case(Q, K, V, scale, n_rep, causal):
    n_hq, S_q, D = Q.shape
    n_hkv = K.shape[0]
    S_k = K.shape[1]
    if n_rep == 1 and n_hkv != n_hq:
        n_rep = n_hq // n_hkv
    Sk_pad = ((S_k + 63) // 64) * 64
    total = n_rep * S_q
    Sq = ((total + 63) // 64) * 64
    rs = np.arange(Sq)
    real = rs < total
    heads = np.where(real, rs // S_q, 0)
    poss = np.where(real, rs % S_q, 0)
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        Qsg = np.zeros((Sq, D), np.float32)
        Qsg[real] = (Q[g * n_rep + heads[real], poss[real]] * float(scale)).astype(np.float32)
        Kt = np.zeros((D, Sk_pad), np.float32)
        Kt[:, :S_k] = K[g].T
        Vp = np.zeros((Sk_pad, D), np.float32)
        Vp[:S_k] = V[g]
        j = np.arange(Sk_pad)[None, :]
        pos_ok = ((j <= poss[:, None]) if (causal and S_q > 1)
                  else np.ones((Sq, Sk_pad), bool))
        allowed = real[:, None] & (j < S_k) & pos_ok
        M = np.where(allowed, 0.0, -30.0)
        Oref = _ref_attention(
            Qsg.astype(bf16).astype(np.float32),
            Kt.astype(bf16).astype(np.float32),
            Vp.astype(bf16).astype(np.float32),
            M.astype(bf16).astype(np.float32))
        out[g * n_rep] = Oref[:S_q]
    return out


def sign_hist(x):
    x = np.asarray(x)
    n = x.size
    return (f"n={n} pos={int((x>0).sum())} "
            f"neg={int((x<0).sum())} |max|={float(np.abs(x).max()):.3f}")


def main():
    orig = npu_gemm.npu_attention_fused_impl
    layer_calls = {}
    cur_layer = [0]

    def hook_fused(Q, K, V, scale, n_rep=1, causal=False):
        layer_calls.setdefault(cur_layer[0], []).append(
            dict(Q=Q.copy(), K=K.copy(), V=V.copy(),
                 scale=scale, n_rep=n_rep, causal=causal))
        return orig(Q, K, V, scale, n_rep, causal)

    npu_gemm.npu_attention_fused_impl = hook_fused

    orig_disp = npu_gemm.npu_attention
    layer_counter = [0]

    def hook_disp(Q, K, V, scale, n_rep=1, causal=False):
        cur_layer[0] = layer_counter[0]
        r = orig_disp(Q, K, V, scale, n_rep, causal)
        layer_counter[0] += 1
        return r

    npu_gemm.npu_attention = hook_disp

    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    n_layers = len(e.layers)
    print(f"built engine: {n_layers} layers, npu_available={e.npu_available}\n", flush=True)

    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    list(e.generate(PROMPT, {"max_tokens": 6, "temperature": 0.0, "top_k": 1}))

    print(f"recorded {len(layer_calls)} layers\n", flush=True)
    worst = []
    for li, calls in sorted(layer_calls.items()):
        for ci, c in enumerate(calls):
            Q, K, V = c["Q"], c["K"], c["V"]
            n_hq, S_q, D = Q.shape
            ref = _ref_case(Q, K, V, c["scale"], c["n_rep"], c["causal"])
            npu = orig(Q, K, V, c["scale"], c["n_rep"], c["causal"])
            d = np.abs(ref.astype(np.float64) - npu.astype(np.float64))
            d[:, ~np.ones(S_q, bool), :] = 0.0
            dmax = float(d.max())
            worst.append((li, ci, dmax, S_q, n_hq, c["n_rep"], c["causal"]))
            if dmax > 1e-3:
                print(f"  L{li} ci{ci} S_q={S_q} nhq={n_hq} n_rep={c['n_rep']} "
                      f"causal={c['causal']} dmax={dmax:.6g}  "
                      f"Q[{sign_hist(Q)}]\n", flush=True)

    worst.sort(key=lambda t: -t[2])
    print("--- top divergences (ref-vs-fused-NPU) ---", flush=True)
    for li, ci, dmax, Sq, nhq, nrep, ca in worst[:8]:
        print(f"  L{li} ci{ci} S_q={Sq} nhq={nhq} n_rep={nrep} causal={ca} "
              f"dmax={dmax:.6g}")

    print(f"\nmax dmax = {worst[0][2]:.6g} at L{worst[0][0]} ci{worst[0][1]} "
          f"S_q={worst[0][3]}", flush=True)
    print("=> dmax~0: NPU agrees w/ ref; diff is wrapper/accumulator path", flush=True)
    print("=> dmax>>0: NPU kernel mishandles that layer's data", flush=True)


if __name__ == "__main__":
    main()
