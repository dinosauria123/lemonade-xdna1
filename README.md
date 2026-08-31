# ⚡ lemonade-xdna1

### The XDNA1 NPU (Ryzen AI "Phoenix" / "Hawk Point") as a Lemonade / OpenAI-compatible backend

`lemonade-xdna1` runs the first‑generation AMD **XDNA1 NPU** in millions of Ryzen
laptops through **Lemonade** (or any OpenAI‑compatible client) with **100% open source**
— an OpenAI‑compatible shim on port **8901** that offloads GEMMs to the NPU, plus a
small real LLM engine (Qwen2.5‑0.5B).

- **Host:** AMD Ryzen AI 9 (HXT) 5700U / Ryzen 7 8845HS, XDNA 1st gen, 10 TOPS
- **Stack:** XRT + `amdxdna` driver + firmware + mlir‑aie (IRON) + Peano, all on Linux
- **NPU backend id:** `xdna1` (`gemm_offload: true`); also `xdna1-mock` (wiring smoke test)
- **Runtime:** Python 3.14 venv (`shimenv`), torch (CPU), transformers

> Note: this is a fork of [**Scottcjn/open-xdna**](https://github.com/Scottcjn/open-xdna).
> See the [Fork](#fork) section.

## Fork

`lemonade-xdna1` is a fork of [**Scottcjn/open-xdna**](https://github.com/Scottcjn/open-xdna)
(the open-source XDNA1 bring-up project). The fork packages that code as a **Lemonade
cloud backend**: an OpenAI-compatible HTTP shim (`server/xdna_openai_server.py`) plus a
real 0.5B engine, registered against lemonade so the NPU is reachable on port **13305**.

---

## Install

Everything below was verified on this box. Upstream `scripts/setup_iron.sh` assumes
Python 3.13 with broken paths, so it is **not** used — `shimenv` (Python 3.14) pulls in
the AIE wheels directly.

```bash
cd ~
git clone https://github.com/dinosauria123/lemonade-xdna1.git
cd lemonade-xdna1
```

### 1. NPU driver + firmware (apt)

```bash
sudo apt update && sudo apt install -y amdxdna-dkms xrt-base xrt-base-dev \
    libxrt-npu2 libxrt-utils
lsmod | grep -iE 'amdxdna|amd_pmf|amdxcp|amd_atl'      # all must show
ls -la /dev/accel/accel0                               # accel class (new XRT)
```

### 2. LLVM 21 (AIE compiler / JIT engine)

```bash
sudo apt install -y llvm-21
ls -d /usr/lib/llvm-21 && /usr/lib/llvm-21/bin/llvm-objcopy --version
```

### 3. shimenv (Python 3.14) + AIE wheels

```bash
python3.14 -m venv shimenv
./shimenv/bin/pip install --upgrade pip wheel
./shimenv/bin/pip install mlir_aie \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4
./shimenv/bin/pip install llvm-aie \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
./shimenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./shimenv/bin/pip install transformers numpy ml_dtypes py-spy
```

### 4. Set the XRT environment (no `source`, which can hang)

```bash
export XILINX_XRT=/opt/xilinx/xrt
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH
export PATH=/usr/lib/llvm-21/bin:$PATH
source ./shimenv/bin/activate
```

Verify before starting the shim:

```bash
./shimenv/bin/python -c "import aie, aie.iron; print('aie.iron OK')"
./shimenv/bin/python -c "import sys; sys.path.insert(0,'server'); import npu_gemm; print(npu_gemm.available())"
# → available()= True   (False ⇒ check XILINX_XRT / LD_LIBRARY_PATH / PYTHONPATH)
```

## Run

```bash
cd /path/to/lemonade-xdna1
nohup python server/xdna_openai_server.py > server/xdna.log 2>&1 &
```

First run JIT-compiles every GEMM/attention shape (a few minutes). Test the wiring:

```bash
curl -s -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"xdna1-mock","messages":[{"role":"user","content":"hi"}]}'
```

Register with Lemonade (optional):

```bash
lemonade cloud install xdna \
  --base-url http://127.0.0.1:8901/v1 --allow-insecure-http
lemonade list    # → xdna.qwen2.5-0.5b  Yes  cloud
```

Talk to the NPU via Lemonade on port 13305:

```bash
curl -s -X POST http://127.0.0.1:13305/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"xdna.qwen2.5-0.5b","messages":[
        {"role":"user","content":"Say hi in 3 words."}],"stream":false}'
```

The model is downloaded from HuggingFace on first use
(`~/.cache/huggingface/hub/`).

## ⚡ Performance: for LLM decode the NPU is ~250× slower than the CPU

Verified on a real Ryzen AI 9 (XDNA1, 5700U), decoding Qwen2.5-0.5B:

| Config | Env | Decode |
|:-------|:----|:-------|
| **CPU only** | `XDNA_NPU_GEMM=0` | **~35 tok/s** ← fastest on this box |
| NPU GEMM (default) | — | ~0.14 tok/s (**~250× slower**) |
| NPU GEMM + NPU attention | `XDNA_NPU_ATTENTION=1` | ~0 tok/s (unusable, 26 s for 0 tokens) |
| NPU prefill only | `XDNA_NPU_GEMM_PREFILL_ONLY=1` | 11–21 tok/s (still slower than CPU) |

**Root cause:** each XDNA1 NPU run has ~50 ms of XRT/DRM dispatch overhead, while a
decode GEMM (M=1 matvec) is only µs — so the overhead wins by four orders of magnitude.
The NPU pays off at **large M** (long prefill / batching) or CNN/vision work, not
single‑token text decode.

**Recommendation:** run the shim with `XDNA_NPU_GEMM=0` for LLM decode on this box;
keep the NPU for CNN/vision or large‑M GEMMs. (`XDNA_NPU_ATTENTION_IMPL=fused` is
disabled by default — it diverges on the real model.)

## ⚠️ Gotchas

- **Single NPU context** — while the shim holds it, other processes get
  `DRM_IOCTL_AMDXDNA_CREATE_HWCTX failed (err=-22)` and fall back to CPU *silently*;
  `npu_gemm.available()` still returns True. Run at most ONE NPU consumer at a time.
- **Init / non‑shell start** — a shim started from init (PID 1) shows *Empty reply*
  to `/health` even while LISTENing; check `server/xdna.log` and restart.
- **Startup hang** — set `XDNA_NPU_WARMUP=0` and the first request triggers JIT.
- **One NPU per machine** — do not run this shim and any other NPU client simultaneously.
