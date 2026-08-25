import sys, time
import numpy as np
import npu_gemm

# Capture whether a fallback happened
import io
from contextlib import redirect_stderr

# Realistic Qwen2.5-0.5B dims: n_heads_q=14, n_heads_kv=2, head_dim=64
nh_q, nh_kv, D = 14, 2, 64
n_rep = nh_q // nh_kv
scale = 1.0 / (D ** 0.5)
S_q, S_k = 1, 128  # decode: 1 query, 128 cached keys

np.random.seed(42)
Q = (np.random.randn(nh_q, S_q, D) * 0.5).astype(np.dtype("bfloat16"))
K = (np.random.randn(nh_kv, S_k, D) * 0.5).astype(np.dtype("bfloat16"))
V = (np.random.randn(nh_kv, S_k, D) * 0.5).astype(np.dtype("bfloat16"))

print("=== NPU path (NPU free) ===")
print("available:", npu_gemm.available(), "| error:", npu_gemm.error())
# warm the context
_ = npu_gemm.matmul_bf16(Q[0], K[0])

err = io.StringIO()
t0 = time.time()
with redirect_stderr(err):
    out = npu_gemm.npu_attention_core(Q, K, V, scale, n_rep, causal=False)
dt = time.time() - t0
fell_back = "fallback" in err.getvalue()
print(f"npu_attention_core: {out.shape} dt={dt*1000:.1f}ms fell_back={fell_back}")
if fell_back:
    print("  stderr:", err.getvalue().strip()[:200])

# CPU reference
def cpu_attn(Q, K, V, scale, n_rep):
    out = np.empty((Q.shape[0], Q.shape[1], Q.shape[2]), dtype=np.float32)
    for h in range(Q.shape[0]):
        q = Q[h].astype(np.float32)
        k = K[h // n_rep].astype(np.float32)
        v = V[h // n_rep].astype(np.float32)
        s = (q @ k.T) * scale
        z = s - s.max(-1, keepdims=True); A = np.exp(z); A /= A.sum(-1, keepdims=True)
        out[h] = A @ v
    return out

ref = cpu_attn(Q, K, V, scale, n_rep)
print(f"finite: {np.isfinite(out).all()}  | max_abs_diff_vs_CPU = {np.abs(out - ref).max():.6f}")
print(f"out[0,:4] = {out[0,:4]}")
print(f"ref[0,:4] = {ref[0,:4]}")
