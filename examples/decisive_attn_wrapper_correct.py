#!/usr/bin/env python3
"""DECISIVE: is the wrapper packing/mask actually buggy, or was EVERY prior
divergence an artifact of a BUGGY host reference harness?

Root cause found in previous harnesses (npu_wrapper_ref / _ref_case):
    out[g*n_rep:(g+1)*n_rep] = P[:S_q] @ Vp      # WRONG: only head-0 rows
P is packed with head g occupying rows [g*S_q:(g+1)*S_q). P[:S_q] only
extracts head 0, so head g>=1 gets head-0's Q/K softmax weights. That alone
produces the large per-layer dmax SKAIL attributed to a "wrapper bug":

    verify_attn_harness.py: harness-vs-CHOST-GTA dmax=1.99 (DIVERGE);
                          perhead-vs-TORCH dmax=0.007 (EXACT)  -> harness wrong.

This test: (1) prove the CORRECTED reference reproduces the model's own
attention operator (torch SDPA, enable_gqa), (2) feed the same REAL Q/K/V into
the real NPU fused kernel, and (3) show npu == corrected_ref == SDPA. If all
three agree on real data, the wrapper packing IS correct and the prior
"overflow / wrapper bug" conclusions were harness artifacts.

NPU stays the ONLY consumer (no other process may hold the XDNA context).
"""
import os, sys, math
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "server"))

os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"

import importlib, engine as engine_mod
importlib.reload(engine_mod)
import numpy as np
import torch
import torch.nn.functional as F
import npu_gemm


def npu_wrapper_ref_correct(Q, K, V, scale, n_rep, causal):
    """Corrected host reference of the fused wrapper.

    Head g occupies packed rows [g*S_q:(g+1)*S_q); rows 0:n_rep of that block
    are real (pos<n_rep). This extracts them correctly (the old harness used
    P[:S_q] which only covered head 0). Returns (n_hq,S_q,D) f32 -- same layout
    as the NPU output.
    """
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
        pos_ok = ((j <= poss[:, None]) if (causal and S_q > 1)
                  else np.ones((Sq, Sk_pad), bool))
        allowed = real[:, None] & (j < S_k) & pos_ok
        M = np.where(allowed, 0.0, -30.0)
        T = Qs @ Kt + M
        z = T - T.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        O = P @ Vp                    # (Sq, D)
        # Unpack EXACTLY as the wrapper does (npu_gemm.py:473-474):
        # out[g*n_rep + heads[real], poss[real]] = O_np[real rows].
        idx = np.where(real)[0]
        oh = g * n_rep + heads[idx]
        out[oh, poss[idx]] = O[idx]
    return out


def torch_sdpa_truth(Qh, Kh, Vh, n_rep, causal, S_q, S_k, D):
    """torch SDPA on unpacked GHA = the model's true attention output.

    Qh/Kh/Vh: (H,S,D). Returns (H,S,D) -- same layout as the NPU output."""
    out = np.zeros(Qh.shape, np.float32)
    for g in range(Kh.shape[0]):
        qg = torch.from_numpy(Qh[g].astype(np.float32))          # (n_rep,S_q,D)
        kg = torch.from_numpy(Kh[g].astype(np.float32))          # (S_k,D)
        vg = torch.from_numpy(Vh[g].astype(np.float32))
        mask = None
        if causal and S_q > 1:
            mask = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool), 1)
        o = F.scaled_dot_product_attention(
            qg, kg[None].expand(n_rep, S_k, D).contiguous(),
            vg[None].expand(n_rep, S_k, D).contiguous(),
            attn_mask=mask)
        out[g * n_rep:(g + 1) * n_rep] = o.numpy()
    return out


def main():
    orig = npu_gemm.npu_attention_fused_impl
    calls = []

    def hook(Q, K, V, scale, n_rep=1, causal=False):
        out = orig(Q, K, V, scale, n_rep, causal)
        calls.append(dict(Q=Q.astype(np.float64), K=K.astype(np.float64),
                          V=V.astype(np.float64), scale=float(scale),
                          n_rep=n_rep, causal=bool(causal),
                          npu=out.astype(np.float64)))
        return out

    npu_gemm.npu_attention_fused_impl = hook

    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    print(f"engine: {len(e.layers)} layers, npu_available={e.npu_available}\n")
    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    # capture first 6 tokens of decode+prefill
    toks = list(e.generate(PROMPT, {"max_tokens": 8, "temperature": 0.0, "top_k": 1}))
    print(f"generated: {''.join(toks)!r}  ({len(calls)} fused calls)\n")

    # Compare corrected_ref and torch SDPA against the real NPU kernel, per call.
    worst_npu = 0.0; worst_h = 0.0; worst_call = None
    for i, c in enumerate(calls):
        Q, K, V, scale, causal = c["Q"], c["K"], c["V"], c["scale"], c["causal"]
        n_rep = c["n_rep"]
        # torch SDPA truth on the SAME real unpacked Q/K/V
        truth = torch_sdpa_truth(Q, K, V, n_rep, causal,
                                 Q.shape[1], K.shape[1], Q.shape[2])
        h = npu_wrapper_ref_correct(Q, K, V, scale, n_rep, causal)
        d_np = float(np.abs(c["npu"] - h).max())
        d_th = float(np.abs(h - truth).max())
        worst_call = i
        worst_npu = max(worst_npu, d_np)
        worst_h = max(worst_h, d_th)
        print(f"  call#{i:2d} S_q={Q.shape[1]} n_rep={n_rep} "
              f"npuref-vs-SDPA={d_th:.5f} npu-vs-npuref={d_np:.5f}")
    print(f"\n--- worst: corrected_ref-vs-SDPA={worst_h:.5f} | "
          f"npu-vs-corrected_ref={worst_npu:.5f} ---")
    if worst_h < 1e-3 and worst_npu < 1e-3:
        print("VERDICT: corrected_ref == torch SDPA == NPU kernel on real "
              "inputs -> WRAPPER packing is CORRECT. Prior 'overflow / wrapper "
              "bug' dmax were HARNESS artifacts (head g>=1 mis-extraction).")
    else:
        print("VERDICT: residual divergence remains -- corrected_ref-vs-SDPA="
              f"{worst_h:.5f}, npu-vs-ref={worst_npu:.5f} (investigate further).")


if __name__ == "__main__":
    main()
