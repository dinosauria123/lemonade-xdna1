"""Clean deterministic 3-way comparison (no model, no engine, no loop).

Compares, on a SMALL synthetic problem with KNOWN-correct answer via torch
SDPA:

  1. torch SDPA                (ground truth: model's own operator)
  2. debug_fused_attn._ref_case (numpy reference)
  3. npu_attention_fused_impl   (the NPU kernel)

Inputs per npu_attention_fused_impl contract:
  Q [n_rep, S_q, D] raw, K/V [1, S_k, D] raw, scale = D^-0.5, causal=True
D=64 required by the kernel.
"""
import sys
sys.path.insert(0, '/home/dino/open-xdna')
sys.path.insert(0, '/home/dino/open-xdna/examples')
import numpy as np
import torch
import torch.nn.functional as F
from debug_fused_attn import _ref_case
from server import npu_gemm


def main():
    D, n_rep, S_q, S_k = 64, 3, 5, 7
    scale = D ** -0.5
    torch.manual_seed(1234)
    Q = torch.randn(n_rep, S_q, D).numpy()        # raw queries
    K = torch.randn(1, S_k, D).numpy()            # raw keys
    V = torch.randn(1, S_k, D).numpy()            # raw values

    # ---- (3) NPU fused path (raises on failure -> bug evidence) ----------
    try:
        npu = npu_gemm.npu_attention_fused_impl(
            Q.astype(np.float32), K.astype(np.float32),
            V.astype(np.float32), scale, n_rep=n_rep, causal=True)
    except Exception as e:
        print(f"NPU raised: {type(e).__name__}: {e}")
        npu = None

    # ---- (2) numpy reference ----
    ref = _ref_case(Q.astype(np.float32), K.astype(np.float32),
                    V.astype(np.float32), scale, n_rep=n_rep, causal=True)

    # ---- (1) torch SDPA ground truth ----
    q = torch.tensor(Q, dtype=torch.float32)                    # [n_rep, S_q, D]
    k = torch.tensor(K, dtype=torch.float32).expand(n_rep, S_k, D)
    v = torch.tensor(V, dtype=torch.float32).expand(n_rep, S_k, D)
    mask = torch.ones(S_q, S_k, dtype=torch.bool)
    for i in range(S_q):
        for j in range(S_k):
            mask[i, j] = (j <= i)
    t = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch_out = t.numpy()

    print("shape:", t.shape)
    print(f"torch_out |max| = {np.abs(torch_out).max():.5f}")
    print(f"ref     |max| = {np.abs(ref).max():.5f}")
    if npu is not None:
        print(f"npu     |max| = {np.abs(npu).max():.5f}")

    d_ref_torch = np.abs(ref - torch_out).max()
    print(f"\n[ref vs torch SDPA] dmax = {d_ref_torch:.5f}")
    if npu is not None:
        d_npu_torch = np.abs(npu - torch_out).max()
        d_npu_ref = np.abs(npu - ref).max()
        print(f"[npu vs torch SDPA] dmax = {d_npu_torch:.5f}")
        print(f"[npu vs ref]        dmax = {d_npu_ref:.5f}")
        print(f"\nVERDICT: torch==ref ? {d_ref_torch < 1e-4} ; "
              f"npu==torch ? {d_npu_torch < 1e-4} ; "
              f"npu==ref ? {d_npu_ref < 1e-4}")
        if npu == torch_out and torch_out == ref:
            print("ALL THREE AGREE -> NPU kernel correct (revert is right)")
        elif npu != torch_out:
            print("NPU DISAGREES with torch SDPA -> NPU kernel buggy")
        elif ref != torch_out:
            print("ref disagrees with torch SDPA -> numpy ref is wrong, not NPU")


if __name__ == "__main__":
    main()
