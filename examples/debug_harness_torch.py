#!/usr/bin/env python3
"""Decisive harness-vs-torch decomposition.  NO NPU.  Fast.

Ground truth = torch.scaled_dot_product_attention(enable_gqa=True): this is the
EXACT operator the model computes, so it is authoritative.

Compare the production-wrapper numpy path (harness_ref) directly to torch SDPA
across an isolation ladder, and ablate ONE ingredient at a time to see which
removal makes harness == torch.  This pinpoints the wrapper bug field.  The
mask/scaling/packing in harness_ref mirrors
server/npu_gemm.py:npu_attention_fused_impl verbatim.
"""
import numpy as np
import torch
import torch.nn.functional as F


def harness_ref(Q, K, V, scale, n_rep, causal, ablate=None):
    """Production-wrapper numpy path (mirrors npu_attention_fused_impl)."""
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
        Qs[real] = Q[g * n_rep + heads[real], poss[real]]
        if ablate != "noscale":
            Qs = Qs * float(scale)
        Kt = np.zeros((D, Sk_pad), np.float32); Kt[:, :S_k] = K[g].T
        Vp = np.zeros((Sk_pad, D), np.float32); Vp[:S_k] = V[g]
        j = np.arange(Sk_pad)[None, :]
        pos_ok = ((j <= poss[:, None]) if causal else
                  np.ones((Sq, Sk_pad), bool))
        allowed = real[:, None] & (j < S_k) & pos_ok
        if ablate == "nomask":
            Mfull = np.zeros((Sq, Sk_pad), np.float32)
        else:
            Mfull = np.where(allowed, 0.0, -30.0)
        T = Qs[:total] @ Kt[:D, :S_k] + Mfull[:total, :S_k]
        z = T - T.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        out_unpack = P[:total] @ Vp[:S_k, :]          # (total, D) full rows
        # Faithful unpack (mirror npu_attention_fused_impl oh/poss indexing):
        for i in range(total):
            if real[i]:
                out[g * n_rep + heads[i], poss[i]] = out_unpack[i]
    return out


def torch_ref(Q, K, V, scale, n_rep, causal):
    q = torch.from_numpy(Q.astype(np.float32))
    k = torch.from_numpy(K.astype(np.float32))
    v = torch.from_numpy(V.astype(np.float32))
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                          enable_gqa=True).numpy()


def run_case(name, n_rep, n_hkv, S_q, S_k, causal, D=64):
    n_hq = n_hkv * n_rep
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((n_hq, S_q, D)).astype(np.float32)
    K = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)
    V = rng.standard_normal((n_hkv, S_k, D)).astype(np.float32)
    scale = D ** -0.5
    t = torch_ref(Q, K, V, scale, n_rep, causal)
    full = harness_ref(Q, K, V, scale, n_rep, causal)
    d_full = float(np.abs(full - t).max())
    abl = {
        "noscale": float(np.abs(harness_ref(Q, K, V, scale, n_rep, causal,
                                            "noscale") - t).max()),
        "nomask":  float(np.abs(harness_ref(Q, K, V, scale, n_rep, causal,
                                            "nomask") - t).max()),
    }
    cause = []
    if abl["noscale"] < 1e-4: cause.append("SCALE")
    if abl["nomask"] < 1e-4: cause.append("MASK")
    print(f"{name}\n  harness-vs-torch dmax={d_full:.6f} "
          f"{'OK' if d_full<1e-4 else 'DIVERGE'}  ablation "
          f"noscale={abl['noscale']:.4f} nomask={abl['nomask']:.4f}  "
          f"{'CAUSE:'+','.join(cause) if cause else '(neither)'}")


def main():
    print("=== harness_ref vs torch SDPA (enable_gqa) — direct ===")
    run_case("(A) n_rep=1, single KV, causal", 1, 1, 3, 5, True)
    run_case("(A2) n_rep=1, single KV, non-causal", 1, 1, 3, 5, False)
    run_case("(B1) n_rep=2, 1 KV head, causal", 2, 1, 3, 5, True)
    run_case("(B2) n_rep=2, 1 KV head, non-causal", 2, 1, 3, 5, False)
    run_case("(C) n_rep=3, n_hkv=2, causal (verify case)", 3, 2, 5, 7, True)
    run_case("(C2) n_rep=3, n_hkv=2, non-causal", 3, 2, 5, 7, False)
    run_case("(D) n_rep=7, n_hkv=1, causal (Qwen prefill)", 7, 1, 41, 41, True)
    run_case("(E) n_rep=7, n_hkv=1, decode", 7, 1, 1, 100, False)


if __name__ == "__main__":
    main()
