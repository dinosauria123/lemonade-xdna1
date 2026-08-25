#!/usr/bin/env python3
"""Verify npu_gemm.matmul_bf16 numerical correctness vs numpy on NPU."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import numpy as np
import npu_gemm
from ml_dtypes import bfloat16

print("NPU available:", npu_gemm.available(), npu_gemm.error())

rng = np.random.default_rng(42)
cases = [
    ("proj decode M=1,K=896,N=1152", 1, 896, 1152),
    ("proj prefill M=32,K=896,N=1152", 32, 896, 1152),
    ("attn QKt M=20,K=64,N=64", 20, 64, 64),
    ("attn AV M=20,K=20,N=64", 20, 20, 64),
]
for name, M, K, N in cases:
    a = rng.standard_normal((M, K)).astype(np.float32).astype(bfloat16)
    w = rng.standard_normal((N, K)).astype(np.float32).astype(bfloat16)
    out = npu_gemm.matmul_bf16(a, w)  # f32 [M,N]
    ref = a.astype(np.float32) @ w.astype(np.float32).T
    # cosine over flattened
    cos = float((out.ravel() @ ref.ravel()) / (np.linalg.norm(out.ravel()) * np.linalg.norm(ref.ravel()) + 1e-9))
    rel = np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-9)
    print(f"{name:32s} shape={out.shape}  cos={cos:.5f}  rel_err={rel:.4f}")
print("DONE")
