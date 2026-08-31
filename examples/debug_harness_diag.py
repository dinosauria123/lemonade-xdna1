#!/usr/bin/env python3
"""Diagnose WHY harness (npu_wrapper_ref) diverges from torch SDPA on the
synthetic inputs verify_attn_harness.py used.  NO NPU contact, fast.

Isolation ladder:
  (A) single KV head, n_rep=1            -> isolates mask/scale/ordering
  (B) n_rep GQA, explicit repeat         -> isolates the GQA packing
  (C) reproduce verify's n_rep=3 n_hkv=2 -> the DIVERGE case

For each: build the model's TRUE GQA output by an INDEPENDENT explicit
per-head loop (not torch SDPA) and compare harness vs that.  Also print
harness vs torch SDPA.  Pinpoints mask / scale / GQA-packing.
"""
import numpy as np
import torch
import torch.nn.functional as F


def harness_ref(Q, K, V, scale, n_rep, causal):
    n_hq, S_q, D = Q.shape
    n_hkv = K.shape[0]
    S_k = K.shape[1]
    Sk_pad = ((S_k + 63) // 64) * 64
    total = n_rep * S_q
    Sq = ((total + 63) // 64) * 64
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
        pos_ok = (j <= poss[:, None]) if (causal and S_q > 1) \
            else np.ones((Sq, Sk_pad), bool)
        allowed = real[:, None] & (j < S_k) & pos_ok
        M = np.where(allowed, 0.0, -30.0)
        T = Qs @ Kt + M
        z = T - T.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        out[g * n_rep:(g + 1) * n_rep] = P[:S_q] @ Vp
    return out


def explicit_gqa_ref(Q, K, V, scale, n_rep, causal, D):
    """Independent ground truth: explicit per-head loop.
    Q (n_hq,S_q,D), K/V (n_hkv,S_k,D). head h attends to KV head h//n_rep.
    Returns (n_hq,S_q,D)."""
    n_hq, S_q, Dd = Q.shape
    n_hkv = K.shape[0]
    S_k = K.shape[1]
    q = torch.from_numpy(Q.astype(np.float32))   # (n_hq,S_q,D)
    k = torch.from_numpy(K.astype(np.float32))   # (n_hkv,S_k,D)
    v = torch.from_numpy(V.astype(np.float32))
    out = np.zeros((n_hq, S_q, Dd), np.float32)
    for h in range(n_hq):
        qh = q[h][None, :, :].expand(1, S_q, D)     # (1,S_q,D)
        kh = k[h // n_rep][None, :, :]              # (1,S_k,D)
        vh = v[h // n_rep][None, :, :]
        mask = None
        if causal and S_q > 1:
            mask = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool), 1)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        out[h] = o[0].numpy()
    return out


def run_case(name, n_rep, n_hkv, S_q, S_k, causal, D=64):
    n_hq = n_hkv * n_rep
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((n_hq, S_q, D)).astype(np.float32)
    K = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)
    V = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)
    scale = D ** -0.5
    h = harness_ref(Q, K, V, scale, n_rep, causal)
    g = explicit_gqa_ref(Q, K, V, scale, n_rep, causal, D)
    d_hg = float(np.abs(h - g).max())
    t = F.scaled_dot_product_attention(
        torch.from_numpy(Q.astype(np.float32)),
        torch.from_numpy(K.astype(np.float32)),
        torch.from_numpy(V.astype(np.float32)),
        is_causal=causal, enable_gqa=True).numpy()
    d_th = float(np.abs(t - g).max())
    print(f"{name} | harness-vs-EXPLICIT-GQA dmax={d_hg:.6f} "
          f"{'OK' if d_hg<1e-4 else 'DIVERGE'} | torch-vs-EXPLICIT-GQA "
          f"dmax={d_th:.6f} {'OK' if d_th<1e-4 else 'DIVERGE'}")
    return d_hg


def main():
    print("=== isolation ladder (independent explicit GQA as truth) ===")
    run_case("(A) single KV head n_rep=1", 1, 1, 3, 5, True)
    run_case("(A2) single KV head non-causal", 1, 1, 3, 5, False)
    run_case("(B) n_rep=2 GQA, 2 kv heads", 2, 1, 3, 5, True)
    run_case("(C) verify n_rep=3 n_hkv=2", 3, 2, 5, 7, True)
    run_case("(C2) verify non-causal", 3, 2, 5, 7, False)


if __name__ == "__main__":
    main()
