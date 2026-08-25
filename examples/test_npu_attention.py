#!/usr/bin/env python3
"""Verify npu_gemm.npu_attention_core correctness (GQA) vs numpy reference."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import numpy as np
import npu_gemm
from ml_dtypes import bfloat16

rng = np.random.default_rng(7)
nh_q, nh_kv, S, D = 14, 2, 64, 64  # Qwen2.5-0.5B GQA shape
scale = 1.0 / np.sqrt(D)

# bf16 inputs
Q = rng.standard_normal((nh_q, S, D)).astype(np.float32).astype(bfloat16)
K = rng.standard_normal((nh_kv, S, D)).astype(np.float32).astype(bfloat16)
V = rng.standard_normal((nh_kv, S, D)).astype(np.float32).astype(bfloat16)

out = npu_gemm.npu_attention_core(Q, K, V, scale, n_rep=nh_q // nh_kv)

# numpy reference (with GQA repeat)
n_rep = nh_q // nh_kv
Krep = np.repeat(K, n_rep, axis=0)
Vrep = np.repeat(V, n_rep, axis=0)
ref = np.empty((nh_q, S, D), dtype=np.float32)
for h in range(nh_q):
    sc = (Q[h].astype(np.float32) @ Krep[h].astype(np.float32).T) * scale
    z = sc - sc.max(axis=-1, keepdims=True)
    A = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    ref[h] = A @ Vrep[h].astype(np.float32)

cos = float((out * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
print(f"NPU attention core: shape={out.shape}  cos_vs_numpy={cos:.5f}  (1.0=identical)")
assert cos > 0.99, f"NPU attention output diverged: cos={cos}"
print("ATTENTION CORE OK")
