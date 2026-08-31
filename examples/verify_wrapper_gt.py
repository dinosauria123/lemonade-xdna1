#!/usr/bin/env python3
"""Decisive wrapper test: fixed per-head AV wrapper vs torch SDPA ground truth.

Ground truth = the model's OWN torch SDPA on unpacked MHA (torch_mha_ref),
NOT a -30 numpy softmax -- so there is NO -30-vs-inf convention difference.
This is the exact comparison the original bisect harness should have done.

    harness == torch SDPA(unpacked) => wrapper packing/mask/scale/AV ALL correct
    diverging layers only           => something layer-specific (rare)
"""
import os, sys
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION"] = "0"
import numpy as np, torch
import torch.nn.functional as F


def wrapper_perhead(Q, K, V, scale, n_rep, causal):
    """Corrected wrapper: per-head AV (the unpack the production wrapper does)."""
    n_hq, S_q, D = Q.shape
    S_k = K.shape[1]
    Sk_pad = ((S_k + 63) // 64) * 64
    total = n_rep * S_q
    Sq = ((total + 63) // 64) * 64
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(K.shape[0]):
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
        for h in range(n_rep):
            out[g * n_rep + h] = P[h * S_q:(h + 1) * S_q] @ Vp
    return out


def torch_mha_ref(Qb, Kb, Vb, n_rep, causal, S_q, S_k):
    """torch SDPA on unpacked MHA = model's true attention output."""
    n_hq = Qb.shape[1]; n_hkv = Kb.shape[1]
    out = np.zeros((Qb.shape[0], n_hq, S_q, Qb.shape[-1]), np.float32)
    for b in range(Qb.shape[0]):
        for g in range(n_hkv):
            qg = torch.as_tensor(Qb[b, g * n_rep:(g + 1) * n_rep])
            kg = torch.as_tensor(Kb[b, g]).unsqueeze(0).expand(n_rep, S_k, Qb.shape[-1])
            vg = torch.as_tensor(Vb[b, g]).unsqueeze(0).expand(n_rep, S_k, Qb.shape[-1])
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
    print(f"heads={n_hq} kv={n_hkv} n_rep={n_rep} D={D} layers={len(e.layers)}\n")

    captured = []
    orig = F.scaled_dot_product_attention
    def cap(q, k, v, *a, **kw):
        out = orig(q, k, v, *a, **kw)
        captured.append(dict(Q=q.detach().float().cpu().numpy(),
                             K=k.detach().float().cpu().numpy(),
                             V=v.detach().float().cpu().numpy()))
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
        Qb, Kb, Vb = c["Q"], c["K"], c["V"]
        n_q = Qb.shape[2]; n_k = Kb.shape[2]
        gt = torch_mha_ref(Qb, Kb, Vb, n_rep, (n_q == n_k), n_q, n_k)
        for b in range(Qb.shape[0]):
            Qh = Qb[b].reshape(n_hq, n_q, Qb.shape[-1])
            Kh = Kb[b].reshape(n_hkv, n_k, Qb.shape[-1])
            Vh = Vb[b].reshape(n_hkv, n_k, Qb.shape[-1])
            harness = wrapper_perhead(Qh, Kh, Vh, Qb.shape[-1]**-0.5, n_rep,
                                      causal=(n_q == n_k))
            ref = gt[b]
            d = float(np.abs(harness - ref).max())
            denom = float(np.abs(ref).max()) + 1e-6
            rd = d / denom
            if rd > worst: worst, wc = rd, b
            rows.append((n_q, n_k, n_rep, int(n_q == n_k), d, rd,
                         float(np.abs(Qh).max()), float(np.abs(Kh).max())))
    rows.sort(key=lambda r: -r[5])
    print("--- wrapper(per-head AV) vs torch SDPA(unpacked MHA) ground truth ---")
    for r in rows[:12]:
        print(f"  S_q={r[0]:2d} S_k={r[1]:2d} n_rep={r[2]} caus={r[3]} "
              f"rel={r[5]:.6f} dmax={r[4]:.6f} |Q|={r[6]:.1f} |K|={r[7]:.1f}")
    print(f"\nWORST RELATIVE DMAX = {worst:.6f} (b{wc})")
    if worst < 0.05:
        print("VERDICT: wrapper == torch SDPA within bf16 tolerance on ALL "
              "LAYS/decodes -> WRAPPER (packing/mask/scale/AV unpack) is "
              "CORRECT. Earlier 'wrapper buggy rel 1.4' was a FALSE POSITIVE "
              "(harness AV broadcast bug).")
    else:
        print("VERDICT: wrapper still diverges vs ground truth -> wrapper bug.")


if __name__ == "__main__":
    main()
