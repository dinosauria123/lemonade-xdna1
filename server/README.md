# XDNA1 OpenAI-compatible shim (Lemonade backend)

Makes the XDNA1 NPU usable through **Lemonade** (or any OpenAI-compatible
client) on open source, without an AMD-proprietary runtime.

Lemonade registers backends at build time (the `LEMON_BACKENDS` CMake
registry), so the practical way to add XDNA1 support without forking the
whole server is to expose an **OpenAI-compatible HTTP endpoint** and register
it with Lemonade's *cloud backend* — which is runtime-configured:

    lemonade cloud install xdna \
        --base-url http://127.0.0.1:8901/v1 \
        --allow-insecure-http --api-key ***

This repo implements that endpoint as a thin Python shim plus a real small
LLM engine with NPU GEMM offload through open-xdna (IRON/XRT).

## Components

| file | role |
|---|---|
| `xdna_openai_server.py` | stdlib-only HTTP server, OpenAI wire protocol (`/v1/models`, `/v1/chat/completions` stream + non-stream), token auth optional, `models.json` registry |
| `engine.py` | Qwen2.5-0.5B-Instruct (bf16, torch CPU) manual-decode engine; wraps all 168 per-layer Linear modules with an NPU dispatch hook |
| `npu_gemm.py` | bf16 GEMM bridge to the XDNA1 NPU via the IRON `single_core` matmul kernel (JIT, per-shape cache) |
| `models.json` | model registry (id, backend, HF model id) |
| `run.sh` | the only supported launcher: sets up llvm/objcopy + XRT + `shimenv` (python 3.14 venv with aie/pyxrt/torch) |
| `test_npu_gemm.py` | numerical self-check of the NPU GEMM vs torch at the model's shapes |

## Why python 3.14 / `shimenv`

The installed XRT (2.21.x) ships `pyxrt` built for **cpython-3.14 only**,
while the stock `ironenv` in `mlir-aie/` is 3.13 — so IRON's `device="npu"`
falls back to a CPU-only tensor class there. `shimenv` is the 3.14 venv with
`aie` (mlir_aie cp314 wheel), `llvm-aie`, `numpy`, `ml_dtypes`, `torch` and
`transformers`.

## NPU path: what actually runs on the NPU

Every per-layer Linear (q/k/v/o/gate/up/down, 7 × 24 layers) is dispatched to
the NPU as a bf16 GEMM through the upstream
`single_core` matrix-multiplication kernel (JIT-compiled by aiecc per shape,
design cached in `~/.npu/cache`).

Hardware-derived limits handled host-side in `npu_gemm.py`:

- **M padding to 64** — the kernel's C-tile grouping requires `M_div_m >= 2`;
  decode (M=1) pads to 64 rows, so the NPU computes 64× more than strictly
  needed. That is the price of "NPU always in the loop";
  `XDNA_NPU_GEMM_PREFILL_ONLY=1` offloads only M≥64 GEMMs instead.
- **K/N chunking at 2048** — AIE DMA burst descriptors cap every BD
  dimension at 64; with the 32×32 tile one kernel call is capped at
  N≤2048 and K≤2048. The 4864-wide FFN projections are split into chunks and
  accumulated in f32 (exact).

`/health` reports how much of the last request ran on the NPU
(`last_request_npu_gemms`, `total_npu_flops_gflop`, JIT counts), so "the NPU
did the work" is verifiable from the API itself.

Honest performance note: XDNA1 is not faster than 16 CPU threads (or the
780M iGPU) at 0.5B dense-matmul scale — the NPU's value is the low-power
floor (~6.6 W) and the non-bijective prune/selection coprocessor design, not
throughput. This shim proves the full Lemonade → NPU silicon path and gives a
hookable place for the prune offload.

Measured on this machine (Qwen2.5-0.5B, 15-token prompt + 14 decode tokens,
every per-layer GEMM on the NPU): **2520 NPU GEMMs, 687 GFLOP, 27.9 s NPU
time → ~24.6 GFLOPS effective** (bf16, M_pad=64, single core). Decode at M=1
pads to the 64-row tile, so it is slower than CPU — see
`XDNA_NPU_GEMM_PREFILL_ONLY` for the fast mode.

## Usage

    # register once (with the running lemond)
    lemonade cloud install xdna --base-url http://127.0.0.1:8901/v1 \
        --allow-insecure-http --api-key ***

    # start the shim (NPU env + shimenv)
    bash server/run.sh

    # use it
    lemonade list | grep xdna
    lemonade run xdna.qwen2.5-0.5b -m "hi"

    # or any OpenAI client pointed at http://127.0.0.1:8901/v1
    # (add Authorization: Bearer *** only if XDNA_OAI_KEY is set)

## Environment

- `XDNA_OAI_HOST` / `XDNA_OAI_PORT` (default 127.0.0.1:8901)
- `XDNA_OAI_KEY` (optional; enables token auth)
- `XDNA_OAI_MODELS` (models.json path)
- `XDNA_NPU_GEMM` (default 1; 0 = CPU-only)
- `XDNA_NPU_GEMM_PREFILL_ONLY` (default 0; 1 = NPU only for M≥64 GEMMs)
- `MLIR_AIE_DIR` (default `~/open-xdna/mlir-aie`)
