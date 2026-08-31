#!/usr/bin/env python3
"""DECISIVE: -30-numpy-ref vs torch SDPA, NO wrapper.

The production NPU softmax kernel (softmax_masked.cc line 70) adds -30 to
masked positions (m[i]? 30.0 : 0.0). torch SDPA uses -inf. These are two
DIFFERENT masking conventions that are numerically close ONLY when the
suppressed entries are far below the row max (exp(-inf)=0 vs exp(T-max-30)≈0).
But if a suppressed key is CLOSE to the row max, -30 keeps a small but nonzero
weight while -inf zeroes it -> real divergence, independent of any wrapper.

This test isolates that convention difference on the REAL prefill Q/K/V.

    numpy_ref ~ torch SDPA  => NO convention difference on this data -> the
                               wrapper-side residual (if any) is the true bug.
    numpy_ref != torch SDPA (big) => the prefill 'divergence' is PURELY the
                               -30-vs-infinity convention; the wrapper is
                               NOT at fault there.
"""
import os, sys
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION"] = "0"
import numpy as np, torch
import torch.nn.functional as F


def npref_30(Q, K, V, scale, n_rep, causal):
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


def torch_sdpa_gt(Qb, Kb, Vb, n_rep, causal, S_q, S_k):
    n_hq, D = Qb.shape[1], Qb.shape[-1]
    n_hkv = Kb.shape[1]
    out = np.zeros((Qb.shape[0], n_hq, S_q, D), np.float32)
    for b in range(Qb.shape[0]):
        for g in range(n_hkv):
            qg = torch.as_tensor(Qb[b, g * n_rep:(g + 1) * n_rep])
            kg = torch.as_tensor(Kb[b, g]).unsqueeze(0).expand(n_rep, S_k, D)
            vg = torch.as_tensor(Vb[b, g]).unsqueeze(0).expand(n_rep, S_k, D)
            attn = None
            if causal and S_q > 1:
                # float -inf additive mask (torch SDPA truth)
                attn = torch.full((S_q, S_k), float("-inf"))
                attn = attn.triu(diagonal=1)
            o = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=attn)
            out[b, g * n_rep:(g + 1) * n_rep] = o.numpy()
    return out


def main():
    import importlib, engine as engine_mod
    importlib.reload(engine_mod)
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    cfg = e.model.config
    n_hq = int(cfg.num_attention_heads); n_hkv = int(cfg.num_key_value_heads)
    n_rep = n_hq // n_hkv
    D = int(cfg.hidden_size // n_hq)
    print(f"heads={n_hq} kv={n_hkv} n_rep={n_rep} D={D}\n")

    captured = []
    orig = F.scaled_dot_product_attention
    def cap(q, k, v, *a, **kw):
        captured.append(dict(Q=q.detach().float().cpu().numpy(),
                             K=k.detach().float().cpu().numpy(),
                             V=v.detach().float().cpu().numpy()))
        return orig(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = cap
    torch.nn.functional.scaled_dot_product_attention = cap
    for m in list(sys.modules.values()):
        try:
            if getattr(m, "scaled_dot_product_attention", None) is orig:
                setattr(m, "scaled_dot_product_attention", cap)
        except Exception:
            pass
    list(e.generate([{"role": "user", "content": "What is 2+2? Answer in one word."}],
                    {"max_tokens": 8, "temperature": 0.0, "top_k": 1}))
    F.scaled_dot_product_attention = orig
    print(f"captured {len(captured)} calls\n")

    rows = []
    for c in captured:
        Qb, Kb, Vb = c["Q"], c["K"], c["V"]
        n_q = Qb.shape[2]; n_k = Kb.shape[2]
        causal = (n_q == n_k)
        for b in range(Qb.shape[0]):
            Qh = Qb[b].reshape(n_hq, n_q, Qb.shape[-1])
            Kh = Kb[b].reshape(n_hkv, n_k, Qb.shape[-1])
            Vh = Vb[b].reshape(n_hkv, n_k, Qb.shape[-1])
            nr = npref_30(Qh, Kh, Vh, Qb.shape[-1]**-0.5, n_rep, causal)
            gt = torch_sdpa_gt(Qb, Kb, Vb, n_rep, causal, n_q, n_k)[b]
            d = float(np.abs(nr - gt).max())
            denom = float(np.abs(gt).max()) + 1e-6
            rows.append((n_q, n_k, int(causal), d, d/denom,
                         float(np.abs(Qh).max()), float(np.abs(Kh).max()),
                         float(np.abs(Vh).max())))
    rows.sort(key=lambda r: -r[4])
    print("--- (-30 numpy ref) vs (torch SDPA -inf)  [NO WRAPPER] ---")
    print("    isolates the -30 vs -inf masking CONVENTION difference only\n")
    for r in rows[:12]:
        print(f"  S_q={r[0]:2d} S_k={r[1]:2d} caus={r[2]} "
              f"rel={r[4]:.5f} dmax={r[3]:.5f} |Q|={r[5]:.1f} |K|={r[6]:.1f} |V|={r[7]:.1f}")
    print(f"\nmax convention dmax = {rows[0][3]:.5f}")
    big = [r for r in rows if r[4] > 0.1]
    if not big:
        print("=> NO meaningful -30-vs-inf convention difference on this data -> "
              "prefill 'divergence' in gt test was CONVENTION ARTIFACT; "
              "wrapper is correct, residual is downstream (kernel).")
    else:
        print(f"=> {len(big)} prefill calls show REAL -30-vs-inf convention diff; "
              "the wrapper is NOT at fault there.\n"
              "   BUT this does NOT prove the wrapper is correct in prefill -- "
              "only that this diff is convention, not wrapper.")


if __name__ == "__main__":
    main()
