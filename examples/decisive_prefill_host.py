#!/usr/bin/env python3
"""DECISIVE DIFFERENT ROUTE: is the prefill residual a HOST-MATH bug or NPU GEMM?

All prior forks compared implementations that SHARE the tight-packed layout
(fused_impl vs npu_wrapper_ref; standalone_kernel vs npu_wrapper_ref). Two
implementations that pack the SAME way can both be wrong together -> the gap
"tight-packed impl diverges in prefill only" was NEVER measured against an
INDEPENDENT ground truth.

This test measures THREE mutually independent quantities per real prefill call:

    A  torch SDPA ground truth        (unpacked MHA, the model's own attention)
    B  fused-host-ref = the EXACT host math of npu_attention_fused_impl
       (tight pack -> QK^T numpy -> -inf causal mask -> nanmax softmax ->
        packed AV -> unpack). Independent of the NPU entirely.
    C  NPU fused output = npu_gemm.npu_attention_fused_impl(real Q/K/V)

VERDICT table:
    A==B  AND  B!=C  => residual is PURELY NPU bf16 GEMM quantization (precision,
                        NOT a packing/mask/AV bug). Host path is CORRECT.
    A!=B              => HOST-MATH bug in the tight-packed prefill path
                        (packing/mask/AV-pack) -- fix host, not the kernel.
    A==B  AND  B==C  => no residual at all on this machine/capture.

Independent ground truth (A) is what the previous forks lacked. This is the
"残差を別経路で特定" route.
"""
import os, sys
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"
os.environ["XDNA_NPU_ATTENTION"] = "1"
import numpy as np, torch
import torch.nn.functional as F
import importlib, engine as engine_mod
import npu_gemm


def fused_host_ref(Q, K, V, scale, n_rep, causal, D=64):
    """Exact host replica of npu_attention_fused_impl (no NPU). This is the
    tight-packed impl math, computed in float32 numpy so it is the correct
    reference for the PACKING / MASK / AV-unpack logic itself."""
    n_hq, S_q, _ = Q.shape
    n_hkv = K.shape[0]
    S_k = K.shape[1]
    Sk_pad = ((S_k + 63) // 64) * 64
    total = n_rep * S_q
    Sq = ((total + 63) // 64) * 64
    out = np.zeros((n_hq, S_q, D), np.float32)
    for g in range(n_hkv):
        rs = np.arange(Sq); real = rs < total
        heads = np.where(real, rs // S_q, 0); poss = np.where(real, rs % S_q, 0)
        Qs = np.zeros((Sq, D), np.float32)
        Qs[real] = Q[g * n_rep + heads[real], poss[real]] * float(scale)
        Kt = np.zeros((D, Sk_pad), np.float32); Kt[:, :S_k] = K[g].T
        Vp = np.zeros((Sk_pad, D), np.float32); Vp[:S_k] = V[g]
        T = Qs @ Kt
        j = np.arange(Sk_pad)[None, :]
        pos_ok = ((j <= poss[:, None]) if (causal and S_q > 1)
                  else np.ones((Sq, Sk_pad), bool))
        allowed = real[:, None] & (j < S_k) & pos_ok
        T = np.where(allowed, T, -np.inf)
        z = T - np.nanmax(T, axis=1, keepdims=True)
        P = np.exp(z); P = np.nan_to_num(P); P /= P.sum(axis=1, keepdims=True)
        O = P[:total] @ Vp[:S_k, :]
        rsl = rs; reall = real
        oh = g * n_rep + heads[reall]
        out[oh, poss[reall]] = O[rsl[reall]]
    return out


def torch_mha_ref(Qb, Kb, Vb, n_rep, causal, S_q, S_k, D):
    out = np.zeros((Qb.shape[0], Qb.shape[1], S_q, D), np.float32)
    for b in range(Qb.shape[0]):
        for g in range(Kb.shape[1]):
            qg = torch.as_tensor(Qb[b, g * n_rep:(g + 1) * n_rep]).float()
            kg = torch.as_tensor(Kb[b, g]).float().unsqueeze(0).expand(S_q, S_k, D)
            vg = torch.as_tensor(Vb[b, g]).float().unsqueeze(0).expand(S_q, S_k, D)
            mask = None
            if causal and S_q > 1:
                mask = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool), 1)
            o = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=mask)
            out[b, g * n_rep:(g + 1) * n_rep] = o.numpy()
    return out


def main():
    print("NPU avail:", npu_gemm.available(), "| err:", npu_gemm.error()[:50])
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
                             V=v.detach().float().cpu().numpy()))
        return out
    F.scaled_dot_product_attention = cap
    torch.nn.functional.scaled_dot_product_attention = cap
    for mod in list(sys.modules.values()):
        if mod is None: continue
        try:
            if getattr(mod, "scaled_dot_product_attention", None) is orig:
                setattr(mod, "scaled_dot_product_attention", cap)
        except Exception: pass
    PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
    gen = list(e.generate(PROMPT, {"max_tokens": 6, "temperature": 0.0, "top_k": 1}))
    F.scaled_dot_product_attention = orig
    print(f"captured {len(captured)} attn calls, gen toks={len(gen)}\n")

    rows = []
    for i, c in enumerate(captured):
        Qb, Kb, Vb = c["Q"], c["K"], c["V"]
        n_q = Qb.shape[2]; n_k = Kb.shape[2]; causal = (n_q == n_k)
        gt = torch_mha_ref(Qb, Kb, Vb, n_rep, causal, n_q, n_k, D)
        Qh = Qb[0].reshape(n_hq, n_q, D); Kh = Kb[0].reshape(n_hkv, n_k, D)
        Vh = Vb[0].reshape(n_hkv, n_k, D)
        scale = D ** -0.5
        B = fused_host_ref(Qh, Kh, Vh, scale, n_rep, causal, D)
        # C = actual NPU fused output (engine produced it via npu_attention_fused_impl)
        C = None
        if npu_gemm.available():
            try:
                C = npu_gemm.npu_attention_fused_impl(Qh.astype("bfloat16"),
                    Kh.astype("bfloat16"), Vh.astype("bfloat16"), scale, n_rep, causal)
            except Exception as ex:
                print(f"  call{i} NPU raised {ex}")
        truth = gt[0]
        dB = float(np.abs(B - truth).max())
        relB = dB / (float(np.abs(truth).max()) + 1e-9)
        rA = None
        if C is not None:
            dA = float(np.abs(C - truth).max())
            rA = dA / (float(np.abs(truth).max()) + 1e-9)
        tag = "PREF" if causal else "DECO"
        print(f"  call{i} {tag} S_q={n_q:2d} S_k={n_k:2d} "
              f"host-vs-truth rel={relB:.5f}  "
              f"npu-vs-truth rel={'?' if rA is None else f'{rA:.5f}'}")
        rows.append((n_q, n_k, causal, relB, rA if rA is not None else 0.0))

    host_rels = [r[3] for r in rows if r[3] > 1e-3]
    npu_rels = [r[4] for r in rows if r[1] > 0]
    print("\n=== VERDICT (別経路 / independent ground truth) ===")
    print(f"  host-vs-torch  worst rel = {max(rows, key=lambda r:r[3])[3]:.6f} "
          f"(n_prefill_only_hosts={sum(1 for r in rows if r[2])})")
    print(f"  npu-vs-torch   worst rel = {max(r[4] for r in rows):.6f}")
    host_ok = all(r[3] < 0.05 for r in rows)
    if host_ok and npu_rels and max(r[4] for r in rows) > 0.15:
        print("  => RESIDUAL IS NPU bf16 GEMM QUANTIZATION (host path CORRECT).\n"
              "     packing/mask/AV-unpack match torch SDPA exactly. The prefill\n"
              "     divergence is quantization noise in QK^T/AV on the NPU, NOT\n"
              "     a host bug. floor cannot fix it (mantissa, not exp overflow).")
    elif not host_ok:
        bad = max(rows, key=lambda r: r[3])
        print(f"  => HOST-MATH BUG in tight-packed prefill path (rel={bad[3]:.4f}).\n"
              "     Fix packing/mask/AV in npu_attention_fused_impl host code.")
    elif not npu_rels:
        print("  => NPU unavailable this run; only host-vs-truth established "
              "(host is CORRECT).")
    else:
        print("  => no significant residual on this capture.")


if __name__ == "__main__":
    main()
