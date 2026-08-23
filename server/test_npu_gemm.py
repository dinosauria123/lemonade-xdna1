#!/usr/bin/env python3
"""Numerical self-check: NPU bf16 GEMM vs torch bf16 reference.

Runs at the actual Qwen2.5-0.5B GEMM shapes. Prints cosine similarity and max
abs error per shape. Cosine > 0.999 and small max-abs (relative to input scale)
means the NPU path is numerically sound for LLM use."""
import os
import sys

import numpy as np
import torch
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npu_gemm  # noqa: E402

SHAPES = [
    # (M, K, N) — q/o proj, k/v proj, gate/up proj, down proj
    (32, 896, 896),
    (32, 896, 128),
    (32, 896, 4864),
    (32, 4864, 896),
]

rng = np.random.default_rng(0)
ok = True
print(f"npu_available={npu_gemm.available()} ({npu_gemm.error() or 'ok'})")
for M, K, N in SHAPES:
    a = (rng.standard_normal((M, K)) * 0.5).astype(bfloat16)
    w = (rng.standard_normal((N, K)) * 0.5).astype(bfloat16)

    out_npu = npu_gemm.matmul_bf16(a, w)  # f32 [M, N]

    a_t = torch.from_numpy(np.ascontiguousarray(a).astype(np.float32)).to(torch.bfloat16)
    w_t = torch.from_numpy(np.ascontiguousarray(w).astype(np.float32)).to(torch.bfloat16)
    out_ref = (a_t @ w_t.T).float().numpy()

    cos = float(
        (out_npu * out_ref).sum()
        / (np.linalg.norm(out_npu) * np.linalg.norm(out_ref) + 1e-12)
    )
    max_abs = float(np.abs(out_npu - out_ref).max())
    scale = float(np.abs(out_ref).max())
    status = cos > 0.999
    ok &= status
    print(
        f"  [{ 'PASS' if status else 'FAIL' }] M={M:<4} K={K:<5} N={N:<5} "
        f"cos={cos:.6f} max_abs={max_abs:.4f} (ref scale {scale:.1f}) "
        f"rel={max_abs / (scale + 1e-9):.2e}"
    )

print("RESULT:", "ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
