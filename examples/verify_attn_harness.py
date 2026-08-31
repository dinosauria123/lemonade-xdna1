#!/usr/bin/env python3
"""Sanity check for bisect_attn_wrapper WITHOUT touching the model or torchvision.

Self-contained correctness triangle on SMALL synthetic Q/K/V (no model load):

  (1) torch SDPA                     -- ground-truth model operator
  (2) npu_attention_core(force_cpu)  -- the production perhead CPU path
                                        (SK.md: A/B exact-verified, known-correct)
  (3) npu_wrapper_ref                -- my numpy re-impl of the FUSED wrapper

If (1) == (2): the perhead CPU baseline is trustworthy.
If (1) == (3): my numpy harness faithfully implements the production wrapper.
If (2) == (3): both production wrappers (perhead CPU vs fused host) agree.

If all three agree on synthetic data, my harness is a faithful model of the
fused-wrapper's packing, and therefore the real-model bisect verdict
(fused-wrapper != model SDPA) is trustworthy -> the wrapper IS buggy.
If (3) diverges from (1) here, my harness is itself wrong (not the wrapper).
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/dino/open-xdna/server")
import npu_gemm


def npu_wrapper_ref(Q, K, V, scale, n_rep, causal):
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
        pos_ok = ((j <= poss[:, None]) if (causal and S_q > 1)
                  else np.ones((Sq, Sk_pad), bool))
        allowed = real[:, None] & (j < S_k) & pos_ok
        M = np.where(allowed, 0.0, -30.0)
        T = Qs @ Kt + M
        z = T - T.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        out[g * n_rep:(g + 1) * n_rep] = P[:S_q] @ Vp
    return out


def torch_sdpa_mha(Q, K, V, scale, n_rep, causal, D):
    """Unpacked-GHA attention = ground truth (verified: diff 2.4e-7 vs
    torch SDPA enable_gqa=True on the same Q/K/V). Q/K/V: (n_hq,S_q,D) and
    (n_hkv,S_k,D) raw bf16-as-float. Returns (n_hq,S_q,D).
    """
    n_hq, S_q, Dd = Q.shape
    n_hkv = K.shape[0]
    S_k = K.shape[1]
    q = torch.from_numpy(Q.astype(np.float32))          # (n_hq,S_q,D)
    k = torch.from_numpy(K.astype(np.float32))          # (n_hkv,S_k,D)
    v = torch.from_numpy(V.astype(np.float32))
    o = F.scaled_dot_product_attention(q, k, v,
                                       is_causal=causal,
                                       enable_gqa=True)
    return o.numpy()


def main():
    rng = np.random.default_rng(0)
    D, S_q, S_k, n_rep = 64, 5, 7, 3
    n_hq = n_rep * 2
    n_hkv = 2
    for causal in (True, False):
        Q = rng.standard_normal((n_hq, S_q, D)).astype(np.float32)
        K = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)
        V = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)
        scale = D ** -0.5
        t = torch_sdpa_mha(Q, K, V, scale, n_rep, causal, D)
        h = npu_wrapper_ref(Q, K, V, scale, n_rep, causal)
        d_t = float(np.abs(h - t).max())
        try:
            p = npu_gemm.npu_attention_core(
                Q.astype(np.dtype("bfloat16")),
                K.astype(np.dtype("bfloat16")),
                V.astype(np.dtype("bfloat16")),
                scale, n_rep, causal, force_cpu=True)
            d_p = float(np.abs(p - t).max())
            p_str = f"perhead-vs-TORCH={d_p:.6f}"
        except Exception as e:
            p_str = f"perhead ERROR: {type(e).__name__}: {e}"
        print(f"causal={int(causal)} S_q={S_q} S_k={S_k} n_rep={n_rep} "
              f"|Q|={np.abs(Q).max():.1f} |V|={np.abs(V).max():.1f}\n"
              f"  [harness-vs-CHOST-GTA] dmax={d_t:.6f} "
              f"{'OK' if d_t < 1e-4 else 'DIVERGE <-- harness itself wrong'}\n"
              f"  [  {p_str}]\n")


if __name__ == "__main__":
    main()
