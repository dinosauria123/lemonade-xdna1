"""Pinpoint: does np_attention_fused match torch SDPA on the SAME inputs?"""
import sys
sys.path.insert(0, "/home/dino/open-xdna")
sys.path.insert(0, "/home/dino/open-xdna/server")
import numpy as np
import torch
import torch.nn.functional as F
import npu_gemm


def small_causal_problem(h=3, s=5, d=4, n_rep=1):
    """Random MHA packed as (n_rep*s, d) and as (n_rep, s, d)."""
    torch.manual_seed(0)
    Qp = torch.randn(h, s, d)              # packed (n_rep, s, d)
    Kp = torch.randn(h, s, d)
    Vp = torch.randn(h, s, d)
    scale = d ** -0.5
    # torch SDPA ground truth (real MHA)
    Q = Qp.reshape(1, h, s, d).permute(0, 2, 1, 3).reshape(h, s, d)  # (h,s,d)
    o = F.scaled_dot_product_attention(Q, Q, Q, attn_mask=torch.triu(torch.ones(s,s)-1,1).bool() if False else None)
    # simpler: do MHA properly below
    return Qp, Kp, Vp, scale


def torch_mha(Q, K, V, scale, causal):
    h, s, d = Q.shape
    q = Q.reshape(1, h, s, d).permute(0, 2, 1, 3)
    k = K.reshape(1, h, s, d).permute(0, 2, 1, 3)
    v = V.reshape(1, h, s, d).permute(0, 2, 1, 3)
    mask = None
    if causal and s > 1:
        m = torch.triu(torch.ones(s, s, dtype=torch.bool), 1)
        mask = m
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).squeeze(0)


def run():
    torch.manual_seed(42)
    h, s, d = 3, 5, 4
    scale = d ** -0.5
    Q = torch.randn(h, s, d)
    K = torch.randn(h, s, d)
    V = torch.randn(h, s, d)
    Qp = (Q * scale)

    out_np = npu_gemm.npu_attention_fused(Qp.numpy(), K.numpy(), V.numpy(),
                                          scale, causal=True)
    out_ref = torch_mha(Q, K, V, scale, causal=True).numpy()
    dmax = np.abs(out_np - out_ref).max()
    print(f"np_attention_fused vs torch SDPA (causal h={h} s={s} d={d}):")
    print(f"  npu   |max|={np.abs(out_np).max():.4f}")
    print(f"  torch |max|={np.abs(out_ref).max():.4f}")
    print(f"  dmax  ={dmax:.4f}")
    print(f"  MATCH={'YES' if dmax < 1e-4 else 'NO  <-- BUG'}")


if __name__ == "__main__":
    run()
