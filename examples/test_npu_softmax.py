#!/usr/bin/env python3
"""Verify npu_gemm.softmax_bf16 runs on the NPU and matches numpy softmax."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import numpy as np
import npu_gemm
from ml_dtypes import bfloat16

print("NPU available:", npu_gemm.available(), npu_gemm.error())

rng = np.random.default_rng(3)
for n in (64, 128, 256, 512):
    x = rng.standard_normal((4, n)).astype(np.float32)
    xb = x.astype(np.dtype(bfloat16))
    out = npu_gemm.softmax_bf16(xb, axis=-1)
    # numpy reference (bf16 round-trip to mimic NPU path)
    z = xb.astype(np.float32)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    ref = e / e.sum(axis=-1, keepdims=True)
    cos = float((out * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
    rows_ok = np.allclose(out.sum(axis=-1), 1.0, atol=1e-2)
    print(f"n={n:4d}  shape(out)={out.shape}  rows_sum_to_1={rows_ok}  cos_vs_ref={cos:.5f}")
print("DONE")
