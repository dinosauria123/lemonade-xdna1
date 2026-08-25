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

# IRON symbols used by both the GEMM path (single_core) and the softmax path
# (transform_parallel). Imported eagerly so the @iron.jit signature annotations
# in softmax_bf16 (In/Out/CompileTime) and the transform_parallel call resolve
# at module import time (they are referenced inside the JIT-decorated closure).
from aie.iron import CompileTime, In, Out  # noqa: E402
from aie.iron.algorithms import transform_parallel  # noqa: E402

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
    # The single_core kernel tiles by _TILE (32), so the tiling dims K and N
    # must be multiples of 32. Pad (don't raise) so non-conforming shapes --
    # e.g. attention QK^T / AV where N == sequence length -- still run on the
    # NPU; M is padded to _PAD for the same reason.
    K_pad = _ceil_pad(K)
    N_pad = _ceil_pad(N)
    M_pad = _ceil_pad(M)

    a2 = a
    if M_pad != M or K_pad != K:
        a2 = np.zeros((M_pad, K_pad), dtype=a.dtype)
        a2[:M, :K] = a
    w2 = w
    if N_pad != N or K_pad != K:
        w2 = np.zeros((N_pad, K_pad), dtype=w.dtype)
        w2[:N, :K] = w

    key = (M_pad, K_pad, N_pad)
    first = key not in stats["shapes"]
    # out is allocated at the PADDED size because the kernel writes [M_pad, *]
    # tiles; we slice the true [M, N] back at the end.
    out = np.zeros((M_pad, N_pad), dtype=np.float32)
    with _lock:
        stats["shapes"][key] = stats["shapes"].get(key, 0) + 1
        stats["gemms"] += 1
        stats["flops"] += 2.0 * M_pad * K_pad * N_pad
        t0 = time.perf_counter()
        for k0 in range(0, K_pad, _MAX_CHUNK):
            k1 = min(k0 + _MAX_CHUNK, K_pad)
            ac = a2[:, k0:k1]
            if not ac.flags.contiguous:
                ac = ac.copy()
            for n0 in range(0, N_pad, _MAX_CHUNK):
                n1 = min(n0 + _MAX_CHUNK, N_pad)
                wc = w2[n0:n1, k0:k1]
                if not wc.flags.contiguous:
                    wc = wc.copy()
                c = _single_gemm(ac, wc)  # f32 [M_pad, n1-n0]
                out[:, n0:n1] += c[:M_pad, :(n1 - n0)]
        stats["run_s"] += time.perf_counter() - t0
        if first:
            stats["compile_calls"] += 1
    return out[:M, :N]


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


# ---------------------------------------------------------------------------
# Attention inner-loop GEMMs on the NPU.
#
# The projected Q/K/V/O linears are already NPU-offloaded via _wrap_linear.
# The attention CORE -- scores = Q.K^T/sqrt(d), A = softmax(scores),
# out = A.V -- ran in torch (CPU) SDPA. QK^T and AV are [S,S]/[S,D] matmuls:
# the bulk of attention FLOPs. We route them through matmul_bf16 (the same NPU
# kernel as the linears) so the whole per-layer compute except softmax/RoPE/
# Layernorm runs on the NPU.
#
# Softmax stays on CPU: the IRON bf16_softmax kernel loops over the full vector
# internally, so IRON transform() needs tile_size == input_size for correctness
# -> one JIT compile per distinct sequence length (seconds each). Softmax is
# O(S) per row and negligible next to the GEMMs, so CPU is fine.
# ---------------------------------------------------------------------------

def npu_attention_core(Q, K, V, scale, n_rep=1, causal=False):
    """NPU attention core with optional GQA repeat-kv.

    Q: [n_heads_q, S_q, D] bf16. K, V: [n_heads_kv, S_k, D] bf16 (n_heads_kv
    * n_rep == n_heads_q). scale: 1/sqrt(head_dim). Returns [n_heads_q, S_q, D]
    f32 on the NPU path, else CPU. Never raises -- falls back to CPU
    matmul+softmax.

    QK^T uses matmul_bf16(Q, K): Q [S_q,D], K [S_k,D] -> Q @ K^T = [S_q, S_k].
    AV uses matmul_bf16(A, V.T): A [S_q,S_k], V.T [D,S_k] -> A @ (V.T)^T =
    A @ V = [S_q, D]. (Passing V directly would mismatch K: A is [S_q,S_k]
    (K=S_k) but V is [S_k,D] (K=D); only the transpose shares K between a, w.)

    NOTE: during decode S_q (query len) != S_k (KV cache len); slice scores
    to [S_q, S_k] and the AV result to [S_q, D]. When causal=True (prefill,
    no KV cache yet) apply a causal mask so position i only attends to j <= i,
    matching transformers' SDPA is_causal behavior.
    """
    nh_q, S_q, D = Q.shape
    nh_kv = K.shape[0]
    S_k = K.shape[1]
    if n_rep == 1 and nh_kv != nh_q:
        n_rep = nh_q // nh_kv
    out = np.empty((nh_q, S_q, D), dtype=np.float32)
    use_npu = available()
    # Causal mask: query i may attend to key j only if j <= i. For decode
    # (S_q=1) this is a no-op since the single query sees only past keys.
    if causal and S_q > 1:
        cmask = np.triu(np.ones((S_q, S_k), dtype=np.float32), k=1) * -1e9
    else:
        cmask = None
    for h in range(nh_q):
        q = Q[h]
        k = K[h // n_rep] if n_rep > 1 else K[h]
        v = V[h // n_rep] if n_rep > 1 else V[h]
        if use_npu:
            try:
                scores = matmul_bf16(q, k)[:S_q, :S_k] * scale   # [S_q, S_k] f32
                if cmask is not None:
                    scores = scores + cmask
                A = _softmax_rows(scores)                        # [S_q, S_k] f32 (CPU)
                A_bf = A.astype(np.dtype("bfloat16"))
                out[h] = matmul_bf16(A_bf, v.T.copy())[:S_q, :D]  # [S_q, D] f32
                continue
            except Exception as e:  # noqa: BLE001
                global _attn_warned
                if not _attn_warned:
                    _attn_warned = True
                    import sys
                    print(f"[engine] NPU attention fallback: {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
        # CPU fallback
        scores = (q.astype(np.float32) @ k.astype(np.float32).T) * scale
        if cmask is not None:
            scores = scores + cmask
        A = _softmax_rows(scores)
        out[h] = A @ v.astype(np.float32)
    return out


def _softmax_rows(x):
    """Row-wise softmax (CPU), x: [..., n] f32 -> same shape f32."""
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# NPU softmax (experimental / for the test-suite).
#
# The IRON bf16_softmax kernel loops over the full vector internally, so IRON
# transform() requires tile_size == input_size for correctness -> one JIT
# compile per distinct row width. Kept here so the NPU softmax path is
# exercised; the real attention engine does softmax on CPU (it is O(S) per
# row and negligible next to the GEMMs).
# ---------------------------------------------------------------------------
_sm_cache = {}
_kernels_softmax = None


def _get_kernels_softmax():
    global _kernels_softmax
    if _kernels_softmax is None:
        from aie.iron import kernels
        _kernels_softmax = kernels.softmax
    return _kernels_softmax


def softmax_bf16(x, axis=-1):
    """Row-wise bf16 softmax on the NPU; matches np softmax (bf16 round-trip).

    x: [..., n] bf16. Returns [..., n] bf16. softmax is applied along `axis`
    (default last). tile_size must equal the reduction length, so this JITs
    one kernel per distinct last-dim size (cached in _sm_cache).
    """
    import aie.iron as iron
    from ml_dtypes import bfloat16

    x = np.ascontiguousarray(x)
    if x.dtype != np.dtype("bfloat16"):
        x = x.astype(np.dtype("bfloat16"))
    n0 = x.shape[-1]
    n = n0
    # The IRON softmax kernel computes independently per 1024-element tile and
    # requires the reduction length to be a multiple of
    # tile_size (1024) × num_columns (4) × num_channels (1) = 4096. Pad the last
    # dim up to a multiple of 4096 with -inf. Softmax of -inf is 0, so the
    # padding contributes nothing to the per-row normalization and the true row
    # (sliced back below) is bit-identical to softmax over n0 elements.
    p = ((n + 4095) // 4096) * 4096
    if p > n:
        pad = np.full(x.shape[:-1] + (p - n0,),
                      np.finfo(np.float32).min, dtype=np.float32)
        x = np.concatenate([x, pad], axis=-1).astype(x.dtype)
    n = p
    # The kernel takes a single flattened vector (like ml/softmax: a_in is a
    # 1-D vector of `size` elements). Process each row independently so the
    # per-row softmax is correct regardless of the leading-batch shape.
    flat = x.reshape(-1, n)
    a_t = iron.tensor(flat.reshape(-1), dtype=bfloat16, device="npu")
    b_t = iron.zeros_like(a_t)

    fn = _sm_cache.get(n)
    if fn is None:
        size = n

        def _softmax(a_in: In, b_out: Out,
                     *, size: CompileTime[int] = size,
                     num_channels: CompileTime[int] = 1):
            return transform_parallel(
                _get_kernels_softmax()(tile_size=1024),
                np.ndarray[(size,), np.dtype[bfloat16]],
                tile_size=1024,
                num_channels=num_channels,
                pass_size_to_kernel=True,
            )
        fn = iron.jit(_softmax)
        _sm_cache[n] = fn

    # One kernel invocation per row (each is exactly `size` = n elements).
    nrows = flat.shape[0]
    for row in range(nrows):
        a_t = iron.tensor(flat[row].reshape(-1), dtype=bfloat16, device="npu")
        b_t = iron.zeros_like(a_t)
        fn(a_t, b_t, size=n, num_channels=1)
        flat[row] = b_t.numpy().copy().reshape(n)

    out = flat[:, :n0].astype(x.dtype)
    return out.reshape(x.shape[:-1] + (n0,))


_attn_warned = False
