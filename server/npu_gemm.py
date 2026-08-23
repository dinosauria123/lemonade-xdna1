#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# open-xdna :: BF16 GEMM on the XDNA1 NPU via the IRON single_core matmul kernel.
#
# The single_core kernel (mlir-aie/programming_examples/basic/matrix_multiplication/
# single_core) is fully shape-parameterized: M/K/N are CompileTime values tiled by
# 32, inputs bf16/i8/i16, outputs bf16/f32/i32. With b_col_maj=1 the weight matrix
# can be passed as-is in [N, K] row-major — which is exactly how transformers stores
# Linear.weight — so no transpose is needed at call time.
#
# First call per shape JIT-compiles (aiecc, a few seconds); IRON caches compiled
# designs per CompileTime signature, so later calls at the same shape are pure
# host<->NPU transfers + kernel run.
#
# Usage:
#   import npu_gemm
#   if npu_gemm.available():
#       out_f32 = npu_gemm.matmul_bf16(a_bf16, w_bf16)   # a:[M,K] w:[N,K] -> [M,N]
#
# Requires (see run.sh): llvm-objcopy on PATH, XRT setup.sh sourced (pyxrt on
# PYTHONPATH, libs on LD_LIBRARY_PATH), and /dev/accel/accel0 accessible.

import os
import sys
import threading
import time

import numpy as np

_TILE = 32   # kernel tile
# The upstream single_core C-tile grouping (rows_per_block=4 -> group dim 2)
# requires M_div_m >= 2, i.e. M_pad >= 64. M_pad=32 crashes in
# TensorTiler2D.group_tiler ("does not divide evenly"). Padding to 64 is the
# minimal M that works with the unmodified kernel.
_PAD = 64

# mlir-aie single_core kernel location (overridable like in the examples)
MLIR_AIE_DIR = os.environ.get(
    "MLIR_AIE_DIR", os.path.expanduser("~/open-xdna/mlir-aie")
)
_SC_DIR = os.path.join(
    MLIR_AIE_DIR, "programming_examples/basic/matrix_multiplication/single_core"
)

_lock = threading.Lock()
_initialized = False
_available = False
_error = ""

stats = {
    "gemms": 0,          # total GEMM calls dispatched to the NPU
    "flops": 0.0,        # total 2*M*K*N
    "compile_calls": 0,  # first (JIT) call per shape
    "run_s": 0.0,        # cumulative kernel+transfer wall time
    "shapes": {},        # (M,K,N) -> count
}


def _init():
    """Import the kernel once; record availability (never raises)."""
    global _initialized, _available, _error
    if _initialized:
        return _available
    _initialized = True
    try:
        if not os.path.isdir(_SC_DIR):
            _error = f"single_core kernel dir not found: {_SC_DIR}"
            return False
        sys.path.insert(0, _SC_DIR)
        import pyxrt  # noqa: F401  (fails if XRT env is not set up)
        # new pyxrt: enumerate_devices() returns a count (int);
        # old pyxrt returned a list of device objects.
        n_dev = pyxrt.enumerate_devices()
        if not isinstance(n_dev, int):
            n_dev = len(n_dev)
        if n_dev == 0:
            _error = "pyxrt imported but 0 NPU devices visible (driver/firmware?)"
            return False
        from single_core import single_core  # noqa: F401  (import test only)
        import aie.iron  # noqa: F401
        _available = True
        _error = ""
    except Exception as e:  # noqa: BLE001 - availability probe
        _error = f"{type(e).__name__}: {e}"
        _available = False
    return _available


def available():
    return _init()


def error():
    return _error


def _ceil_pad(x):
    return ((x + _PAD - 1) // _PAD) * _PAD


# The AIE DMA burst descriptor limits every BD dimension size to [1:64].
# single_core builds BD dims of size N_div_n (pattern repeat over A) and
# K_div_k (stream over B), so both must stay <= 64: with the 32x32 tile that
# caps one kernel call at N <= 2048 and K <= 2048. Larger GEMMs are split on
# the host side into chunks and accumulated in f32 (exact).
_MAX_CHUNK = 2048


def matmul_bf16(a, w, out_dtype=np.float32):
    """C = A @ W^T on the NPU.

    a: [M, K] bf16 (ml_dtypes.bfloat16) numpy array
    w: [N, K] bf16 numpy array (torch Linear.weight layout, row-major)
    returns [M, N] array in out_dtype (f32 default: keeps full accumulation range)

    M is padded to 64; K/N larger than 2048 are chunked (see _MAX_CHUNK) so
    the DMA BD dimensions stay within hardware limits.

    Raises RuntimeError when the NPU is unavailable — callers must fall back.
    """
    if out_dtype != np.float32:
        raise RuntimeError("this bridge only supports f32 output")
    M, K = a.shape
    N = w.shape[0]
    M_pad = _ceil_pad(M)
    if K % _TILE or N % _TILE:
        raise RuntimeError(f"K={K} and N={N} must be multiples of {_TILE}")

    a2 = a
    if M_pad != M:
        a2 = np.zeros((M_pad, K), dtype=a.dtype)
        a2[:M] = a

    key = (M_pad, K, N)
    first = key not in stats["shapes"]
    out = np.zeros((M, N), dtype=np.float32)
    with _lock:
        stats["shapes"][key] = stats["shapes"].get(key, 0) + 1
        stats["gemms"] += 1
        stats["flops"] += 2.0 * M_pad * K * N
        t0 = time.perf_counter()
        for k0 in range(0, K, _MAX_CHUNK):
            k1 = min(k0 + _MAX_CHUNK, K)
            ac = a2[:, k0:k1]
            if not ac.flags.contiguous:
                ac = ac.copy()
            for n0 in range(0, N, _MAX_CHUNK):
                n1 = min(n0 + _MAX_CHUNK, N)
                wc = w[n0:n1, k0:k1]
                if not wc.flags.contiguous:
                    wc = wc.copy()
                c = _single_gemm(ac, wc)  # f32 [M_pad, n1-n0]
                out[:, n0:n1] += c[:M]
        stats["run_s"] += time.perf_counter() - t0
        if first:
            stats["compile_calls"] += 1
    return out


def _single_gemm(a, w):
    """One kernel call: a [M_pad, K], w [N, K], K and N <= _MAX_CHUNK."""
    import aie.iron as iron
    from ml_dtypes import bfloat16
    from single_core import single_core

    M, K = a.shape
    N = w.shape[0]
    a_t = iron.tensor(a.reshape(-1), dtype=bfloat16, device="npu")
    w_t = iron.tensor(w.reshape(-1), dtype=bfloat16, device="npu")
    c_t = iron.zeros(M * N, dtype=np.dtype(np.float32), device="npu")
    single_core(
        a_t, w_t, c_t,
        M=M, K=K, N=N,
        m=_TILE, k=_TILE, n=_TILE,
        dtype_in_str="bf16",
        dtype_out_str="f32",
        b_col_maj=1,
    )
    # .copy() is mandatory: .numpy() is a view of the IRON host buffer, which
    # is freed when c_t (a local of this function) is garbage collected.
    return c_t.numpy().reshape(M, N).copy()
