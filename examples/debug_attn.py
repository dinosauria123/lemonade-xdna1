import sys; sys.path.insert(0, 'server')
import numpy as np
import npu_gemm
from ml_dtypes import bfloat16

assert npu_gemm.available(), npu_gemm.error()
rng = np.random.default_rng(7)
S, D = 35, 64
Q = rng.standard_normal((14, S, D)).astype(np.float32).astype(bfloat16)
K = rng.standard_normal((2, S, D)).astype(np.float32).astype(bfloat16)
V = rng.standard_normal((2, S, D)).astype(np.float32).astype(bfloat16)
print("calling npu_attention_core S=35 ...", flush=True)
o = npu_gemm.npu_attention_core(Q, K, V, 0.125, n_rep=7)
print("OK out.shape=", o.shape, flush=True)
