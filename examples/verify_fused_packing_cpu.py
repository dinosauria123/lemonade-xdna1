#!/usr/bin/env python3
"""CPU-only verification of the fused-attention HOST PACKING/mask/chunking
logic (npu_attention_fused_impl) vs the proven host reference (npu_wrapper_ref).

No NPU needed: we re-implement the SAME packing/mask/chunk math the wrapper
uses and compare to the reference, WITHOUT touching the LUT/kernel.  This
isolates whether the residual prefill divergence is in the HOST packing/wiring
(packing, causal mask, chunk-index unpack) rather than the NPU LUT overflow.

Run:  python examples/verify_fused_packing_cpu.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))


def npu_wrapper_ref(Q, K, V, scale, n_rep, causal):
    """Proven-correct reference (matches torch SDPA)."""
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
        out_unpack = P[:total] @ Vp[:Sk_pad, :]
        for i in range(total):
            if real[i]:
                out[g * n_rep + heads[i], poss[i]] = out_unpack[i]
    return out


def fused_host_unpack(Q, K, V, scale, n_rep, causal, max_blocks=5):
    """Replicate npu_attention_fused_impl's HOST packing + chunk + unpack,
    but use the reference softmax (no LUT) so we test ONLY the wiring."""
    n_hq, S_q, D = Q.shape
    n_hkv = K.shape[0]; S_k = K.shape[1]
    if n_rep == 1 and n_hkv != n_hq:
        n_rep = n_hq // n_hkv
    Sk_pad = ((S_k + 63) // 64) * 64
    total = n_rep * S_q
    Sq = ((total + 63) // 64) * 64
    PAD = 64
    max_rows = max_blocks * PAD

    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        rs = np.arange(Sq)
        real = rs < total
        heads = np.where(real, rs // S_q, 0)
        poss = np.where(real, rs % S_q, 0)
        Qs = np.zeros((Sq, D), np.float32)
        Qs[real] = Q[g * n_rep + heads[real], poss[real]] * float(scale)
        Kt = np.zeros((D, Sk_pad), np.float32); Kt[:, :S_k] = K[g].T
        Vp = np.zeros((Sk_pad, D), np.float32); Vp[:S_k] = V[g]
        j = np.arange(Sk_pad)[None, :]
        pos_ok = ((j <= poss[:, None]) if (causal and S_q > 1)
                  else np.ones((Sq, Sk_pad), dtype=bool))
        allowed = real[:, None] & (j < S_k) & pos_ok
        M = np.where(allowed, 0, 1).astype(np.int8)  # 0 allowed / 1 masked

        # ---- reference softmax (per chunk, mimics kernel: rows of Qs+M) ----
        # For each chunk, compute softmax over allowed entries (masked=-inf).
        # This is the GROUND TRUTH of the packed layout (no LUT).
        row0 = 0
        while row0 < Sq:
            c = min(max_rows, Sq - row0)
            Qs_c = Qs[row0:row0 + c]
            M_c = M[row0:row0 + c]
            T_c = Qs_c @ Kt + (M_c == 1) * (-30.0)
            # masked -> -inf so exp=0
            T_c = np.where(M_c == 1, -np.inf, T_c)
            z = T_c - np.nanmax(T_c, axis=1, keepdims=True)
            P_c = np.exp(z); P_c = np.nan_to_num(P_c)
            P_c /= P_c.sum(axis=1, keepdims=True)
            O_c = P_c @ Vp[:Sk_pad, :]
            rsl = rs[row0:row0 + c]
            reall = real[row0:row0 + c]
            oh = g * n_rep + heads[row0:row0 + c][reall]
            # THE SUSPECT LINE, replicated:
            out[oh, poss[row0:row0 + c][reall]] = O_c[rsl[reall] - row0]
            row0 += c
    return out


def run(name, n_rep, S_q, S_k, causal=True, seed=1, sigma=1.0):
    rng = np.random.default_rng(seed)
    n_hq = n_rep
    q = rng.standard_normal((n_hq, S_q, 64)).astype(np.float32) * sigma
    k = rng.standard_normal((1, S_k, 64)).astype(np.float32) * sigma
    v = rng.standard_normal((1, S_k, 64)).astype(np.float32)
    scale = 64 ** -0.5
    ref = npu_wrapper_ref(q, k, v, scale, n_rep, causal)
    got = fused_host_unpack(q, k, v, scale, n_rep, causal)
    diff = float(np.abs(got - ref).max())
    denom = float(np.abs(ref).max()) + 1e-9
    rel = diff / denom
    cos = float(np.sum(got * ref) / ((np.linalg.norm(got) * np.linalg.norm(ref)) + 1e-30))
    verdict = "DIVERGE" if (rel > 1e-4 or np.isnan(got).any()) else "OK"
    print(f"{name}: dmax={diff:.5f} rel={rel:.2e} cos={cos:.7f} -> {verdict}")
    return verdict == "OK"


def main():
    print("=== decode (S_q=1) ===")
    run("decode S_q=1 Sk=128", 7, 1, 128, causal=False)
    run("decode S_q=1 Sk=128 causal", 7, 1, 128, causal=True)
    print("=== prefill (S_q>1) ===")
    run("prefill S_q=41 Sk=64 causal", 7, 41, 64, causal=True)
    run("prefill S_q=64 Sk=64 causal (chunked)", 7, 64, 64, causal=True)
    run("prefill S_q=41 Sk=100 causal (long ctx)", 7, 41, 100, causal=True)
    run("prefill S_q=64 Sk=100 causal (chunked+long)", 7, 64, 100, causal=True)


if __name__ == "__main__":
    main()
