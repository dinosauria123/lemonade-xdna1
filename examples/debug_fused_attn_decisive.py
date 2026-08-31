#!/usr/bin/env python3
"""Decisive 3-way comparison on ONE layer's REAL tensors + magnitude signature.

For every recorded call, computes:
  - harness_np : independent numpy packed-transformer math (mirrors wrapper)
  - numpy_ref  : debug_fused_attn._ref_case (causal numpy ref)
  - npu_output : real fused-NPU output (recorded per layer)
Then prints pairwise dmax to locate the bug (kernel vs wrapper vs ref), plus
n_rep, D, S_q, S_kv, n_hq, causal, and |Q|/|K|/|V| magnitude.

This replaces all prior placeholder attempts. Runs purely in numpy for the
harness/ref (no NPU needed for the comparison); NPU output is captured once.
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "mlir-aie", "programming_examples", "ml", "attention"))
sys.path.insert(0, os.path.join(ROOT, "..", "mlir-aie", "programming_examples", "ml", "attention"))
sys.path.insert(0, os.path.join(ROOT, "..", "server"))
import numpy as np

XDNA_NPU_ATTENTION = "1"
os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"

from ml_dtypes import bfloat16 as bf16
import importlib, engine as engine_mod, npu_gemm
importlib.reload(engine_mod)


def _ref_attention(Qs, Kt, V, M):
    T = Qs @ Kt + M
    z = T - T.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)) @ V


def _ref_case(Q, K, V, scale, n_rep, causal):
    """Faithful copy from debug_fused_attn.py — the causal numpy reference."""
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
    out = np.zeros((n_hq, Sq, D), np.float32)
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
        Mm = np.where(allowed, 0.0, -30.0)
        Oref = _ref_attention(
            Qsg.astype(bf16).astype(np.float32),
            Kt.astype(bf16).astype(np.float32),
            Vp.astype(bf16).astype(np.float32),
            Mm.astype(bf16).astype(np.float32))
        out[g * n_rep] = Oref[:S_q]
    return out


def harness_np(Q, K, V, scale, n_rep, causal):
    """Independent packed-transformer math — mirrors the wrapper exactly."""
    D = 64
    head = n_rep
    S_q = Q.shape[-1] // (head * D)
    S_kv = K.shape[-1] // (head * D)
    Qs = (Q * scale).reshape(head, D, -1)
    Ks = K.reshape(head, D, -1)
    Vs = V.reshape(head, D, -1)
    P = np.zeros((head, S_q, S_kv), np.float32)
    A = np.zeros((head, S_q, D), np.float32)
    for hi in range(head):
        q = Qs[hi]; ks = Ks[hi]; vs = Vs[hi]
        for j in range(S_q):
            d = np.einsum("d,jd->j", q[:, j], ks) / scale
            P[hi, j] = d
        Pa = np.exp(P[hi] - P[hi].max(axis=1, keepdims=True))
        if causal:
            cm = np.triu(1) if S_q > 1 else np.zeros((S_q, S_kv), bool)
            Pa = Pa * cm
        A[hi] = np.einsum("ij,jd->id", Pa, vs)
    return A.reshape(-1, S_q, D)


def main():
    # Capture NPU output per layer by hooking the impl.
    orig = npu_gemm.npu_attention_fused_impl
    recorded = []
    def hook(Q, K, V, scale, n_rep=1, causal=False):
        recorded.append(dict(Q=Q, K=K, V=V, scale=scale, n_rep=n_rep, causal=causal))
        return orig(Q, K, V, scale, n_rep, causal)
    npu_gemm.npu_attention_fused_impl = hook

    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    n_layers = len(e.layers)
    print(f"built engine: {n_layers} layers, npu_available={e.npu_available}\n")
    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    list(e.generate(PROMPT, {"max_tokens": 6, "temperature": 0.0, "top_k": 1}))
    print(f"recorded {len(recorded)} call signatures\n")

    rows = []
    for (Qs, Ks, Vs, scale, n_rep, causal), outs in recorded.items():
        Q, K, V = Qs, Ks, Vs
        n_hq, S_q, D = Q.shape
        n_hkv = K.shape[0]
        S_k = K.shape[1]
        eff_n_rep = n_rep
        if eff_n_rep == 1 and n_hkv != n_hq:
            eff_n_rep = n_hq // n_hkv
        ref = _ref_case(Q, K, V, scale, eff_n_rep, causal)
        harn = harness_np(Q, K, V, scale, eff_n_rep, causal)
        npu = np.array(outs[-1], np.float64) if outs else np.zeros_like(ref)
        # keep only real rows
        d_ref_npu = np.abs(ref.astype(np.float64) - npu.astype(np.float64))
        d_harn_npu = np.abs(harn.astype(np.float64) - npu.astype(np.float64))
        d_harn_ref = np.abs(harn.astype(np.float64) - ref.astype(np.float64))
        mask = np.ones(S_q, bool)
        rows.append(dict(
            S_q=S_q, S_k=S_k, n_hq=n_hq, n_hkv=n_hkv, eff_n_rep=eff_n_rep,
            causal=causal, D=D,
            d_ref_npu=float(d_ref_npu.max()),
            d_harn_npu=float(d_harn_npu.max()),
            d_harn_ref=float(d_harn_ref.max()),
            qm=float(np.abs(Q).max()), km=float(np.abs(K).max()), vm=float(np.abs(V).max())))
    rows.sort(key=lambda r: r["S_q"])
    for r in rows[:12]:
        print(f"S_q={r['S_q']:3d} n_hq={r['n_hq']:2d} n_rep={r['eff_n_rep']:2d} "
              f"causal={int(r['causal'])} d(ref|npu)={r['d_ref_npu']:.4f} "
              f"d(harn|npu)={r['d_harn_npu']:.4f} d(harn|ref)={r['d_harn_ref']:.4f} "
              f"|Q|={r['qm']:.1f} |K|={r['km']:.1f} |V|={r['vm']:.1f}")
    print("\n--- worst d_ref_npu (numpy-ref vs NPU) ---")
    rows.sort(key=lambda r: -r["d_ref_npu"])
    for r in rows[:8]:
        print(f"  S_q={r['S_q']:3d} n_hq={r['n_hq']:2d} n_rep={r['eff_n_rep']:2d} "
              f"causal={int(r['causal'])} d_ref_npu={r['d_ref_npu']:.4f} "
              f"d_harn_npu={r['d_harn_npu']:.4f} d_harn_ref={r['d_harn_ref']:.4f}")


if __name__ == "__main__":
    main()
