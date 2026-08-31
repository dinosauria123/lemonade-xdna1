#!/usr/bin/env python3
"""ALL-THREE isolation (the one decisive question): for each REAL attention
call compare

    (a) model_truth  -- the model's ACTUAL SDPA output (captured)
    (b) torch_sdpa   -- SDPA recomputed from the SAME captured Q/K/V
    (c) wrapper_ref  -- the production wrapper numpy logic

    a == b  => capture/reshape faithful; model IS torch SDPA.
    c == b  => the wrapper logic is correct vs torch SDPA.
    c == a  => the wrapper reproduces the model.

    If a==b AND c==a, the wrapper is CORRECT and the prior E2E divergence
    is NOT a packing bug. If c==b but c!=a, then a != b: the model's real
    attention uses something torch SDPA does not (custom backend / scaling),
    and the wrapper-vs-model gap is a precision-implementation detail, not
    packing.  NPU is irrelevant here (CPU-only) so it stays a lone consumer.
"""
import os, sys
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION"] = "0"

import numpy as np
import torch
import torch.nn.functional as F


def wrapper_ref(Q, K, V, scale, n_rep, causal):
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
        O = P @ Vp
        idx = np.where(real)[0]
        out[g * n_rep + heads[idx], poss[idx]] = O[idx]
    return out


def torch_sdpa(Q, K, V, n_rep, causal, S_q, S_k, D):
    out = np.zeros(Q.shape, np.float32)
    for g in range(K.shape[0]):
        qg = torch.from_numpy(Q[g].astype(np.float32))
        kg = torch.from_numpy(K[g].astype(np.float32))
        vg = torch.from_numpy(V[g].astype(np.float32))
        mask = None
        if causal and S_q > 1:
            mask = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool), 1)
        o = F.scaled_dot_product_attention(
            qg, kg[None].expand(n_rep, S_k, D).contiguous(),
            vg[None].expand(n_rep, S_k, D).contiguous(), attn_mask=mask)
        out[g * n_rep:(g + 1) * n_rep] = o.numpy()
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

    c_div_b = 0  # model_truth differs from torch_sdpa (capture/reshape?)
    c_div_a = 0  # wrapper differs from model_truth
    worst = 0.0; wc = 0
    for i, c in enumerate(captured):
        Qh = c["Q"][0].reshape(n_hq, c["Q"].shape[2], D)
        Kh = c["K"][0].reshape(n_hkv, c["K"].shape[2], D)
        Vh = c["V"][0].reshape(n_hkv, c["V"].shape[2], D)
        S_q = Qh.shape[1]; S_k = Kh.shape[1]; causal = (S_q == S_k)
        truth = c["truth"][0]
        tsd = torch_sdpa(Qh, Kh, Vh, n_rep, causal, S_q, S_k, D)
        w = wrapper_ref(Qh, Kh, Vh, D ** -0.5, n_rep, causal)
        d_ab = float(np.abs(truth - tsd).max())
        d_cb = float(np.abs(w - tsd).max())
        d_ca = float(np.abs(w - truth).max())
        denom = float(np.abs(truth).max()) + 1e-9
        rel = d_ca / denom
        if rel > worst:
            worst = rel; wc = i
        flag = ""
        if d_ab > 1e-4: flag += " [a!=b capture]"
        if d_cb > 1e-4: flag += " [c!=b wrapper vs SDPA]"
        if d_ca > 1e-4: flag += " [c!=a wrapper vs model]"
        if d_ab > 1e-4: c_div_b += 1
        if d_ca > 1e-4: c_div_a += 1
        print(f"  #{i:2d} S_q={S_q} S_k={S_k} "
              f"a!=b={d_ab:.4f} c!=b={d_cb:.4f} c!=a={d_ca:.4f} "
              f"rel={rel:.4f}{flag}")
    print(f"\n--- WORST wrapper-vs-model rel={worst:.4f} (#{wc}) ---")
    print(f"calls where model_truth != torch_sdpa: {c_div_b}/{len(captured)}")
    print(f"calls where wrapper  != model_truth   : {c_div_a}/{len(captured)}")
    if c_div_a == 0:
        print("VERDICT: wrapper reproduces the model EXACTLY (no packing bug).")
    elif c_div_b == len(captured):
        print("VERDICT: model captured != torch SDPA everywhere -> capture/reshape "
              "is unfaithful; cannot judge wrapper by model truth this way.")
    else:
        print("VERDICT: residual wrapper-vs-model gaps in some prefill layers; "
              "inspect those specific layers.")


if __name__ == "__main__":
    main()
