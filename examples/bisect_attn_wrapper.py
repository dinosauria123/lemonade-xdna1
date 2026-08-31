#!/usr/bin/env python3
"""DECISIVE wrapper bisection WITHOUT the NPU.

Goal (SK.md NPU-next-step 2): put the model's REAL Q/K/V into an EXACT numpy
copy of the production wrapper (npu_attention_fused_impl) and compare to the
model's OWN attention operator (torch SDPA, causal).

    harness == torch SDPA  => my numpy ref is the production wrapper -> the
                              E2E divergence is at the KERNEL/dispatch, and
                              the production wrapper itself is correct.
    harness != torch SDPA  => the production WRAPPER packing/mask is buggy.

No NPU is contacted: we hook Qwen2Attention's forward inputs (Q/K/V), compute
the production-wrapper numpy path and torch SDPA on those exact tensors. This
mirrors decisive_test.py (torch SDPA ground truth on unpacked MHA) but replaces
the captured NPU output with an EXACT host re-implementation of the wrapper, so
the NPU is irrelevant and the verdict pins the wrapper's own correctness.

    harness = Qs[real] = Q[g*n_rep+heads, poss]*scale
              packed to (Sq,D), causal mask (-30 on future / pad / garble),
              softmax(row), out = A @ V, unpacked the same tight-packed way.
    torch SDPA = the model's true attention output for those K/V/Q.
"""
import os, sys
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION"] = "0"   # force torch SDPA (no NPU)

import numpy as np
import torch
import torch.nn.functional as F


def npu_wrapper_ref(Q, K, V, scale, n_rep, causal):
    """EXACT numpy re-implementation of npu_attention_fused_impl (one block,
    no chunking -- the per-block logic is identical). Q/K/V float32, MHA-packed.
    Returns (n_hq, S_q, D) f32 -- same layout as the NPU output.
    """
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


def torch_mha_ref(Qb, Kb, Vb, n_rep, causal, S_q, S_k):
    """torch SDPA on the unpacked MHA = the model's true attention output.
    Qb/Kb/Vb: (batch, n_heads, S_, D) bfloat16-as-float32.
    Returns (batch, n_hq, S_q, D) f32.
    """
    n_hq, D = Qb.shape[1], Qb.shape[-1]
    n_hkv = Kb.shape[1]
    out = np.zeros((Qb.shape[0], n_hq, S_q, D), np.float32)
    for b in range(Qb.shape[0]):
        for g in range(n_hkv):
            qg = Qb[b, g * n_rep:(g + 1) * n_rep]      # (n_rep, S_q, D)
            kg = Kb[b, g].unsqueeze(0).expand(n_rep, S_k, D)
            vg = Vb[b, g].unsqueeze(0).expand(n_rep, S_k, D)
            mask = None
            if causal and S_q > 1:
                mask = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool), 1)
            o = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=mask)
            out[b, g * n_rep:(g + 1) * n_rep] = o.numpy()
    return out


def main():
    import importlib, engine as engine_mod
    importlib.reload(engine_mod)
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    model = e.model
    cfg = model.config
    n_hq = int(cfg.num_attention_heads)
    n_hkv = int(cfg.num_key_value_heads)
    n_rep = n_hq // n_hkv
    D = int(cfg.hidden_size // n_hq)
    print(f"model heads={n_hq} kv={n_hkv} n_rep={n_rep} D={D} "
          f"layers={len(e.layers)} npu_available={e.npu_available}\n")

    # Capture the model's ACTUAL attention core inputs: monkeypatch
    # torch.nn.functional.scaled_dot_product_attention. transformers calls this
    # with the post-RoPE, post-reshape Q/K/V (batch, heads, seq, D). We record
    # the real tensors AND the model's true output (orig sdpa call) so the
    # wrapper ref is compared to the model's own operator on the same inputs.
    captured = []
    orig_sdpa = F.scaled_dot_product_attention
    def sdpa_capture(q, k, v, *args, **kwargs):
        out = orig_sdpa(q, k, v, *args, **kwargs)
        captured.append(dict(Q=q.detach().float().cpu().numpy(),
                             K=k.detach().float().cpu().numpy(),
                             V=v.detach().float().cpu().numpy(),
                             model_out=out.detach().float().cpu().numpy()))
        return out
    # Patch F.scaled_dot_product_attention everywhere it is imported by (a) torch
    # itself and (b) the transformers model, so the patched version is actually
    # executed during generate().
    F.scaled_dot_product_attention = sdpa_capture
    torch.nn.functional.scaled_dot_product_attention = sdpa_capture
    import sys
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, "scaled_dot_product_attention", None) is orig_sdpa:
                setattr(mod, "scaled_dot_product_attention", sdpa_capture)
        except Exception:
            pass

    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    list(e.generate(PROMPT, {"max_tokens": 8, "temperature": 0.0, "top_k": 1}))
    F.scaled_dot_product_attention = orig_sdpa
    print(f"captured {len(captured)} attention calls (real Q/K/V, torch SDPA)\n")

    tol = 0.02
    worst = 0.0; worst_call = None
    rows = []
    for c in captured:
        Qb, Kb, Vb = c["Q"], c["K"], c["V"]
        n_q = Qb.shape[2]      # S_q (query length)
        n_k = Kb.shape[2]      # S_k (key length)
        if c is captured[0]:
            print("captured Q/K/V shapes:", Qb.shape, Kb.shape, Vb.shape,
                  "model_out:", c["model_out"].shape, "\n")
        model_out = c["model_out"]          # (batch,n_hq,S_q,D) model's true out
        for b in range(Qb.shape[0]):
            # Q/K/V are (batch, n_heads, seq, D); Qb[b] is already
            # (n_hq, S_q, D).  Unpack the packed GQA rows.
            Qh = Qb[b].reshape(n_hq, n_q, D)
            Kh = Kb[b].reshape(n_hkv, n_k, D)
            Vh = Vb[b].reshape(n_hkv, n_k, D)
            # Production-wrapper numpy path (single block; identical per-block).
            harness = npu_wrapper_ref(Qh, Kh, Vh, D ** -0.5, n_rep,
                                      causal=(n_q == n_k))
            d = float(np.abs(harness - model_out[b]).max())
            denom = float(np.abs(model_out[b]).max()) + 1e-6
            rd = d / denom
            if rd > worst:
                worst, worst_call = rd, b
            rows.append((b, n_q, n_k, n_rep, int(n_q == n_k), d, rd,
                         float(np.abs(Qh).max()), float(np.abs(Kh).max()),
                         float(np.abs(Vh).max())))
    rows.sort(key=lambda r: -r[6])

    print("--- top relative-divergences (production WRAPPER vs model's torch SDPA) ---")
    for r in rows[:10]:
        print(f"  b{r[0]} S_q={r[1]:2d} S_k={r[2]:2d} "
              f"n_rep={r[3]} caus={r[4]} rel={r[6]:.4f} dmax={r[5]:.4f} "
              f"|Q|={r[7]:.1f} |K|={r[8]:.1f} |V|={r[9]:.1f}")
    print(f"\n--- WORST RELATIVE DMAX = {worst:.4f} (b{worst_call}) ---")
    if worst < tol:
        print("VERDICT: production WRAPPER == model SDPA within bf16 tolerance -> "
              "the wrapper is CORRECT; E2E divergence is downstream "
              "(kernel/dispatch/accumulator).")
    else:
        print("VERDICT: production WRAPPER != model SDPA beyond tolerance -> "
              "the WRAPPER packing/mask/scale is BUGGY (this is the root cause).")


if __name__ == "__main__":
    main()
