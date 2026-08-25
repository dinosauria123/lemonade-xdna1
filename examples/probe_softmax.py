import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron.algorithms import transform_parallel
from aie.iron import kernels
print("probe transform_parallel softmax", flush=True)

rng = np.random.default_rng(0)
n = 1024
x = rng.uniform(-4.0, 8.0, size=(n,)).astype(bfloat16)
a_t = iron.tensor(x.reshape(-1), dtype=bfloat16, device="npu")
b_t = iron.zeros_like(a_t)

@iron.jit
def _sm(a_in, b_out, *, size=1024, num_channels=1):
    return transform_parallel(
        kernels.softmax(tile_size=1024),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024, num_channels=num_channels, pass_size_to_kernel=True,
    )

_sm(a_t, b_t)
ref = kernels.softmax_ref(x, tile_size=1024)
out = b_t.numpy()
cos = float((out*ref).sum()/(np.linalg.norm(out)*np.linalg.norm(ref)+1e-9))
rows = out.sum(axis=-1)
print(f"softmax_n1024 n={n} cos_vs_ref={cos:.5f} rows_sum~={rows[:1]} flush_ok={bool(np.allclose(rows, 1.0, atol=1e-2))}", flush=True)

# rows softmax on [4,1024]
x2 = rng.uniform(-4.0, 8.0, size=(4, 1024)).astype(bfloat16)
a2 = iron.tensor(x2.reshape(-1), dtype=bfloat16, device="npu")
b2 = iron.zeros_like(a2)
_sm(a2, b2)
ref2 = kernels.softmax_ref(x2, tile_size=1024)
out2 = b2.numpy().reshape(4, 1024)
cos2 = float((out2*ref2).sum()/(np.linalg.norm(out2)*np.linalg.norm(ref2)+1e-9))
print(f"softmax_rows4 {out2.shape} cos_vs_ref={cos2:.5f}", flush=True)
