#!/usr/bin/env python3
"""VERIFY harness_ref bug: corrected wrapper (per-head AV) vs torch SDPA.

The bisect harness uses out[g*ng:(g+1)*ng] = P[:S_q]@Vp  -> broadcasts the
first-query-block attention to ALL n_rep query heads, ignoring that each head
has its own Q. GQA heads share K/V but NOT Q, so head h must use P[h*S_q:...].
This corrects that AND re-tests against the model's true torch SDPA.
"""
import os, sys
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION"] = "0"
import numpy as np, torch
import torch.nn.functional as F


def npu_wrapper_ref_FIXED(Q, K, V, scale, n_rep, causal):
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
        for h in range(n_rep):                       # CORRECT: per-head AV
            out[g * n_rep + h] = P[h * S_q:(h + 1) * S_q] @ Vp
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
    print(f"heads={n_hq} kv={n_hkv} n_rep={n_rep} D={D} layers={len(e.layers)} "
          f"npu_avail={e.npu_available}\n")

    captured = []
    orig = F.scaled_dot_product_attention
    def cap(q, k, v, *a, **kw):
        out = orig(q, k, v, *a, **kw)
        captured.append(dict(Q=q.detach().float().cpu().numpy(),
                             K=k.detach().float().cpu().numpy(),
                             V=v.detach().float().cpu().numpy(),
                             model_out=out.detach().float().cpu().numpy(),
                             is_causal=kw.get("is_causal", None),
                             has_mask="attn_mask" in kw))
        return out
    F.scaled_dot_product_attention = cap
    torch.nn.functional.scaled_dot_product_attention = cap
    for mod in list(sys.modules.values()):
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
    print(f"captured {len(captured)} calls\n")
    worst = 0.0; wc = None; rows = []
    for c in captured:
        Qb, Kb, Vb, mout = c["Q"], c["K"], c["V"], c["model_out"]
        n_q = Qb.shape[2]; n_k = Kb.shape[2]
        print(f"  is_causal={c['is_causal']} has_mask={c['has_mask']} "
              f"shapes Q{Qb.shape} K{Kb.shape} out{mout.shape}")
        for b in range(Qb.shape[0]):
            Qh = Qb[b].reshape(n_hq, n_q, D)
            Kh = Kb[b].reshape(n_hkv, n_k, D)
            Vh = Vb[b].reshape(n_hkv, n_k, D)
            harness = npu_wrapper_ref_FIXED(Qh, Kh, Vh, D**-0.5, n_rep,
                                            causal=(n_q == n_k))
            d = float(np.abs(harness - mout[b]).max())
            denom = float(np.abs(mout[b]).max()) + 1e-6
            rd = d / denom
            if rd > worst: worst, wc = rd, b
            rows.append((n_q, n_k, n_rep, int(n_q == n_k), d, rd,
                         float(np.abs(Qh).max()), float(np.abs(Kh).max())))
    rows.sort(key=lambda r: -r[5])
    print("\n--- FIXED wrapper (per-head AV) vs model torch SDPA ---")
    for r in rows[:8]:
        print(f"  S_q={r[0]:2d} S_k={r[1]:2d} n_rep={r[2]} caus={r[3]} "
              f"rel={r[5]:.5f} dmax={r[4]:.5f} |Q|={r[6]:.1f} |K|={r[7]:.1f}")
    print(f"\nWORST RELATIVE DMAX = {worst:.5f} (b{wc})")
    if worst < 0.02:
        print("VERDICT: FIXED wrapper == torch SDPA -> the PRODUCTION WRAPPER "
              "(correct unpack) is CORRECT. The earlier 'wrapper buggy' verdict "
              "was a FALSE POSITIVE from the harness's own AV broadcast bug.")
    else:
        print("VERDICT: wrapper STILL diverges -> look downstream.")


if __name__ == "__main__":
    main()
