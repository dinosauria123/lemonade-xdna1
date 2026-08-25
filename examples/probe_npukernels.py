import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, Out, kernels
from aie.iron.algorithms import transform_parallel
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import npu_gemm

print("NPU avail:", npu_gemm.available(), npu_gemm.error(), flush=True)

# --- 1. single_core matmul on NPU ---
rng = np.random.default_rng(0)
S = 35
a = rng.standard_normal((1, 64)).astype(bfloat16)
w = rng.standard_normal((64, 64)).astype(bfloat16)
t0 = time.perf_counter()
o = npu_gemm.matmul_bf16(a, w)
print(f"[{time.perf_counter()-t0:.2f}s] matmul {a.shape}@{w.shape} -> {o.shape}", flush=True)
ref = a.astype(np.float32) @ w.astype(np.float32).T
cos = float((o.ravel() @ ref.ravel())/(np.linalg.norm(o.ravel())*np.linalg.norm(ref)+1e-9))
print(f"matmul cos_vs_numpy={cos:.5f}", flush=True)

# --- 2. softmax on NPU: rows softmax, [nh, S], S=35 ---
@iron.jit
def _softmax_rows(a_in, b_out, *, size, num_channels):
    return transform_parallel(
        kernels.softmax(tile_size=1024),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024, num_channels=num_channels, pass_size_to_kernel=True,
    )

nh, Sd = 14, 35
x = (rng.standard_normal((nh, Sd))*5 + 20).astype(bfloat16)  # large to test numerical
a_t = iron.tensor(x.reshape(-1), dtype=bfloat16, device="npu")
b_t = iron.zeros_like(a_t)
t0 = time.perf_counter()
_softmax_rows(a_t, b_t, size=x.size, num_channels=1)
print(f"[{time.perf_counter()-t0:.2f}s] softmax NPU done", flush=True)
out = b_t.numpy()
ref2 = kernels.softmax_ref(x, tile_size=1024)
cos2 = float((out*ref2).sum()/(np.linalg.norm(out)*np.linalg.norm(ref2)+1e-9))
rows = out.sum(axis=-1)
print(f"softmax_rows {out.shape} cos_vs_ref={cos2:.5f} rows_sum={rows[:1]}~1={bool(np.allclose(rows,1.0,atol=1e-2))}", flush=True)
