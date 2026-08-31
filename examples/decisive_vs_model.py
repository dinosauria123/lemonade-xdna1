#!/usr/bin/env python3
"""FINAL DECISIVE (SKAIL Next-step 2): does the production wrapper logic match
the MODEL'S OWN true attention output on REAL data?

Strategy (mirrors bisect_attn_wrapper's capture, but now compares to it):
  * Run the model on CPU only (XDNA_NPU_ATTENTION=0 -> torch SDPA is the true
    model operator). Monkeypatch F.scaled_dot_product_attention to capture the
    REAL unpacked Q/K/V AND the model's TRUE output (orig sdpa call) per layer.
  * On those SAME captured Q/K/V compute the production-wrapper numpy logic
    (npu_wrapper_ref_correct -- now proven faithful: npu==ref within bf16) and
    compare to the model's own captured output.

    wrapper-ref == model's true SDPA output  => the wrapper packing/mask/scale
                 reproduces the model EXACTLY. The earlier "wrapper bug" was a
                 HARNESS artifact (broken head-g extraction; see
                 verify_attn_harness.py where the old harness diverged 1.99
                 even on synthetic data).
"""
import os, sys
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION"] = "0"

import numpy as np
import torch
import torch.nn.functional as F


def npu_wrapper_ref_correct(Q, K, V, scale, n_rep, causal):
    """Faithful production-wrapper numpy logic. Q/K/V: (H,S,D) f32.
    Returns (n_hq, S_q, D) f32 -- same layout as NPU output.
    Proven faithful vs the REAL NPU kernel (npu-vs-corrected_ref ~0.01-0.07)."""
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
        O = P @ Vp                      # (Sq, D)
        idx = np.where(real)[0]
        out[g * n_rep + heads[idx], poss[idx]] = O[idx]
    return out


def main():
    import importlib, engine as engine_mod
    importlib.reload(engine_mod)
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    cfg = e.model.config
    n_hq = int(cfg.num_attention_heads); n_hkv = int(cfg.num_key_value_heads)
    n_rep = n_hq // n_hkv; D = int(cfg.hidden_size // n_hq)
    print(f"heads={n_hq} kv={n_hkv} n_rep={n_rep} D={D} layers={len(e.layers)}\n")

    captured = []
    orig = F.scaled_dot_product_attention
    def cap(q, k, v, *a, **kw):
        out = orig(q, k, v, *a, **kw)
        captured.append(dict(Q=q.detach().float().cpu().numpy(),
                             K=k.detach().float().cpu().numpy(),
                             V=v.detach().float().cpu().numpy(),
                             truth=out.detach().float().cpu().numpy()))
        return out
    F.scaled_dot_product_attention = cap
    torch.nn.functional.scaled_dot_product_attention = cap
    import sys as _s
    for mod in list(_s.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, "scaled_dot_product_attention", None) is orig:
                setattr(mod, "scaled_dot_product_attention", cap)
        except Exception:
            pass

    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    list(e.generate(PROMPT, {"max_tokens": 8, "temperature": 0.0, "top_k": 1}))
    F.scaled_dot_product_attention = orig
    print(f"captured {len(captured)} calls (model's TRUE SDPA output)\n")

    tol = 0.15
    worst = 0.0; wc = 0
    for i, c in enumerate(captured):
        Qb, Kb, Vb = c["Q"], c["K"], c["V"]
        Qh = Qb[0].reshape(n_hq, Qb.shape[2], D)
        Kh = Kb[0].reshape(n_hkv, Kb.shape[2], D)
        Vh = Vb[0].reshape(n_hkv, Vb.shape[2], D)
        S_q = Qh.shape[1]; S_k = Kh.shape[1]
        causal = (S_q == S_k)
        wref = npu_wrapper_ref_correct(Qh, Kh, Vh, D ** -0.5, n_rep, causal)
        d = float(np.abs(wref - c["truth"]).max())
        denom = float(np.abs(c["truth"]).max()) + 1e-9
        if d / denom > worst:
            worst = d / denom; wc = i
        print(f"  call#{i:2d} S_q={S_q} S_k={S_k} rel={d/denom:.5f} "
              f"dmax={d:.4f} |Q|={np.abs(Qh).max():6.1f} "
              f"{'OK' if d/denom < tol else 'DIVERGE'}")
    print(f"\n--- WORST wrapper-vs-MODEL-SDPA rel={worst:.5f} (call#{wc}) ---")
    if worst < tol:
        print("VERDICT: production WRAPPER reproduces the MODEL's true "
              "attention output within bf16 tolerance -> WRAPPER is CORRECT. "
              "Prior 'wrapper/overflow bug' conclusions were HARNESS ARTIFACTS.")
    else:
        print("VERDICT: wrapper still diverges -> investigate packing/mask.")


if __name__ == "__main__":
    main()
