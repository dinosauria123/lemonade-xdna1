#!/usr/bin/env python3
"""DECISIVE FORK: isolate attention-kernel correctness for REAL decode/prefill.

Reproduce, on the STANDALONE fused kernel (mm_attention_llm) with its own
numpy reference (same mask), the REAL Q/K/V captured from the actual
generation. Compare kernel-vs-numpy:

    STANDALONE decode OK, E2E decode diverges -> bug is in the DECODE PATH
       (KV-cache packing in engine.py / npu_gemm impl), NOT the attention
       kernel itself.
    STANDALONE decode diverges -> the attention KERNEL body is buggy (overflow
       floor / masking / chunking).

This is the decisive fork the previous per-call comparison could not name.
"""
import os, sys
os.environ["XDNA_NPU_ATTENTION"] = "1"
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
sys.path.insert(0, "/home/dino/open-xdna/server")
import numpy as np
import importlib, engine as engine_mod
import npu_gemm

# ---- standalone fused kernel + its own numpy ref (same mask) ----
sys.path.insert(0,
    "/home/dino/open-xdna/mlir-aie/programming_examples/ml/attention")
from mm_attention_llm import fused_llm_attention   # noqa: E402
from aie.iron import str_to_dtype   # noqa: E402
bf16 = str_to_dtype("bf16")


def npu_wrapper_ref(Q, K, V, scale, n_rep, causal):
    n_hq, S_q, D = Q.shape
    n_hkv = K.shape[0]; S_k = K.shape[1]
    Sk = ((S_k + 63) // 64) * 64
    total = n_rep * S_q; Sq = ((total + 63) // 64) * 64
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        rs = np.arange(Sq); real = rs < total
        heads = np.where(real, rs // S_q, 0); poss = np.where(real, rs % S_q, 0)
        Qs = np.zeros((Sq, D), np.float32)
        Qs[real] = Q[g * n_rep + heads[real], poss[real]] * float(scale)
        Kt = np.zeros((D, Sk), np.float32); Kt[:, :S_k] = K[g].T
        Vp = np.zeros((Sk, D), np.float32); Vp[:S_k] = V[g]
        j = np.arange(Sk)[None, :]
        posok = (j <= poss[:, None]) if (causal and S_q > 1) else np.ones((Sq, Sk), bool)
        allowed = real[:, None] & (j < S_k) & posok
        M = np.where(allowed, 0.0, -30.0)
        T = Qs @ Kt + M
        z = T - T.max(axis=1, keepdims=True)
        P = np.exp(z); P /= P.sum(axis=1, keepdims=True)
        uu = P[:total] @ Vp[:S_k, :]
        for i in range(total):
            if real[i]:
                out[g * n_rep + heads[i], poss[i]] = uu[i]
    return out


def standalone_kernel(Qh, Kh, Vh, scale, n_rep, causal, n_hkv, D=64):
    """Run the REAL fused kernel per KV group on these Q/K/V and return the
    unpacked result (mirrors engine: for group g, row r -> head g*n_rep+r//S_q)."""
    n_hq, S_q, _ = Qh.shape
    n_hkv = Kh.shape[0]; S_k = Kh.shape[1]
    Sk = ((S_k + 63) // 64) * 64
    total = n_rep * S_q; Sq = ((total + 63) // 64) * 64
    rs = np.arange(Sq); real = rs < total
    heads = np.where(real, rs // S_q, 0); poss = np.where(real, rs % S_q, 0)
    bf16 = str_to_dtype("bf16")
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        Qs = np.zeros((Sq, D), np.float32)
        rs = np.arange(Sq)
        real_r = rs[real]
        ph = real_r // S_q
        pp = real_r % S_q
        Qs[real] = Qh[(g * n_rep + ph), pp] * float(scale)
        Kt = np.zeros((D, Sk), np.float32); Kt[:, :S_k] = Kh[g].T
        Vp = np.zeros((Sk, D), np.float32); Vp[:S_k] = Vh[g]
        j = np.arange(Sk)[None, :]
        posok = (j <= poss[:, None]) if (causal and S_q > 1) else np.ones((Sq, Sk), bool)
        allowed = real[:, None] & (j < S_k) & posok
        M_i8 = np.where(allowed, 0, 1).astype(np.int8)
        Qs_t = iron_tensor(Qs.reshape(-1).astype(bf16))
        Kt_t = iron_tensor(Kt.reshape(-1).astype(bf16))
        V_t = iron_tensor(Vp.reshape(-1).astype(bf16))
        M_t = iron_tensor(M_i8.reshape(-1))
        O_t = iron_zeros(Sq * D)
        fused_llm_attention(Qs_t, Kt_t, V_t, M_t, O_t,
                            Sq=Sq, Sk=Sk, D=D, element_type=bf16)
        O_np = O_t.numpy().copy().reshape(Sq, D)
        for r in range(total):
            out[g * n_rep + (r // S_q)] = O_np[g * Sq + r]
    return out


def iron_tensor(a):
    import aie.iron as iron
    return iron.tensor(a.reshape(-1), dtype=bf16, device="npu")


def iron_zeros(n):
    import aie.iron as iron
    return iron.zeros(n, dtype=bf16, device="npu")


def main():
    importlib.reload(engine_mod)
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    cfg = e.model.config
    n_hq = int(cfg.num_attention_heads); n_hkv = int(cfg.num_key_value_heads)
    n_rep = n_hq // n_hkv
    print(f"heads={n_hq} kv={n_hkv} n_rep={n_rep}\n", flush=True)

    calls = []
    orig_fused = npu_gemm.npu_attention_fused_impl
    def hook_fused(Q, K, V, scale, n_rep=1, causal=False):
        r = orig_fused(Q, K, V, scale, n_rep, causal)
        calls.append(dict(Q=Q.astype(np.float32).copy(),
                          K=K.astype(np.float32).copy(),
                          V=V.astype(np.float32).copy(),
                          out=r.copy(), causal=bool(causal)))
        return r
    npu_gemm.npu_attention_fused_impl = hook_fused
    list(e.generate([{"role": "user", "content": "What is 2+2? Answer in one word."}],
                    {"max_tokens": 4, "temperature": 0.0, "top_k": 1}))
    npu_gemm.npu_attention_fused_impl = orig_fused
    print(f"captured {len(calls)} fused calls\n", flush=True)

    dec_ok = dec_fail = 0
    pref_ok = pref_fail = 0
    for i, c in enumerate(calls):
        S_q = c["Q"].shape[1]; causal = c["causal"]
        # standalone kernel on EXACT same inputs (per group)
        k_out = standalone_kernel(c["Q"], c["K"], c["V"],
                                  1.0 / (c["Q"].shape[-1] ** 0.5),
                                  n_rep, causal, n_hkv, c["Q"].shape[-1])
        ref = npu_wrapper_ref(c["Q"], c["K"], c["V"],
                              1.0 / (c["Q"].shape[-1] ** 0.5),
                                  n_rep, causal)
        d = float(np.abs(k_out - c["out"]).max())      # kernel vs engine-fused
        d2 = float(np.abs(k_out - ref).max())          # kernel vs numpy ref
        rel2 = d2 / (float(np.abs(ref).max()) + 1e-6)
        verdict = "OK" if rel2 < 0.05 else "DIVERGE"
        if causal:
            if rel2 < 0.05: pref_ok += 1
            else: pref_fail += 1
            tag = "PREF"
        else:
            if rel2 < 0.05: dec_ok += 1
            else: dec_fail += 1
            tag = "DECO"
        print(f"  call{i} {tag} S_q={S_q:2d} kernel-vs-npu rel={rel2:.4f} "
              f"kernel-vs-engine d={d:.4f} {verdict}")
    print(f"\n=== STANDALONE KERNEL (same mask, same inputs) ===")
    print(f"  DECODE: OK={dec_ok} DIVERGE={dec_fail}")
    print(f"  PREFILL: OK={pref_ok} DIVERGE={pref_fail}")
    if dec_fail and dec_ok == 0:
        print("  => ATTENTION KERNEL BODY IS BUGGY (reproduced standalone at "
              "decode). Root cause inside kernel (overflow floor / masking).")
    elif dec_ok and dec_fail == 0:
        print("  => ATTENTION KERNEL BODY IS CORRECT standalone. E2E decode "
              "divergence must come from the DECODE PATH (KV-cache/packing), "
              "NOT the kernel.")
    else:
        print("  => split: kernel correct for some shapes, buggy for others -> "
              "shape-dependent bug (chunk boundary / Sk padding).")


if __name__ == "__main__":
    main()
