import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import numpy as np
from ml_dtypes import bfloat16
import npu_gemm

t0 = time.perf_counter()
ok = npu_gemm.available()
print(f"[{time.perf_counter()-t0:.1f}s] NPU avail={ok} err={npu_gemm.error()}", flush=True)
if not ok:
    sys.exit(1)

# single_core matmul: [1,64] @ [1152,64]^T
rng = np.random.default_rng(0)
a = rng.standard_normal((1, 64)).astype(bfloat16)
w = rng.standard_normal((1152, 64)).astype(bfloat16)
t0 = time.perf_counter()
o = npu_gemm.matmul_bf16(a, w)
print(f"[{time.perf_counter()-t0:.2f}s] matmul_bf16 {o.shape} cos_ok={float((o.ravel()@ref_ok) if False else 0.0):.5f}", flush=True)
ref = a.astype(np.float32) @ w.astype(np.float32).T
cos = float((o.ravel() @ ref.ravel())/(np.linalg.norm(o.ravel())*np.linalg.norm(ref)+1e-9))
print(f"matmul cos_vs_numpy={cos:.5f}", flush=True)
