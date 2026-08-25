#!/usr/bin/env bash
# open-xdna :: run the OpenAI shim with the full NPU environment.
#
# This is the only supported way to start the server when NPU GEMM offload is
# wanted: the IRON runtime needs llvm-objcopy on PATH (aiecc JIT), XRT's
# setup.sh for pyxrt + libs, and the shimenv python (3.14 — the only version
# with a pyxrt build for this XRT install).
#
# Usage:  bash server/run.sh
# Env overrides: XDNA_OAI_HOST, XDNA_OAI_PORT, XDNA_OAI_KEY, XDNA_OAI_MODELS,
#                XDNA_NPU_GEMM (default 1), XDNA_NPU_ATTENTION (default 1),
#                XDNA_NPU_GEMM_PREFILL_ONLY (default 0)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/usr/lib/llvm-21/bin:${PATH}"        # llvm-objcopy for aiecc JIT
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1   # XILINX_XRT, LD_LIBRARY_PATH, PYTHONPATH
source "${HERE}/../shimenv/bin/activate"          # python 3.14 + torch + mlir_aie

export XDNA_OAI_HOST="${XDNA_OAI_HOST:-127.0.0.1}"
export XDNA_OAI_PORT="${XDNA_OAI_PORT:-8901}"
export XDNA_OAI_MODELS="${XDNA_OAI_MODELS:-${HERE}/models.json}"
export XDNA_NPU_GEMM="${XDNA_NPU_GEMM:-1}"
export XDNA_NPU_ATTENTION="${XDNA_NPU_ATTENTION:-1}"
export XDNA_NPU_GEMM_PREFILL_ONLY="${XDNA_NPU_GEMM_PREFILL_ONLY:-0}"

exec python "${HERE}/xdna_openai_server.py"
