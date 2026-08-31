# XDNA1 NPU 手順書 (open-xdna shim 経由)

他のPC (AMD Ryzen AI 9 / 8845HS 系、XDNA 1st gen) で open-xdna の
OpenAI-compat shim を立てて NPU を使えるようにするための完全手順。

この手順は実際に動作確認済みの内容に基づく。upstream の `scripts/setup_iron.sh`
は Python 3.13 前提・パスが破損しているため**使用せず**、
shimenv (Python 3.14) 配下の `aie` / `mlir_aie` / `llvm-aie` を直接使う方式。

---

## 0. 前提ハードウェア / OS

| 項目 | 値 |
|---|---|
| CPU | AMD Ryzen 9 (HXT) 5700U または Ryzen 7 8845HS w/ Radeon 780M |
| NPU | XDNA 1st gen, 10 TOPS (Phoenix Hawk Point) |
| OS | Ubuntu 25.10+ (26.04 で動作確認) |
| カーネル | `amdxdna-dkms` (amdxdna / amd_pmf / amdxcp / amd_atl モジュール) |

**NPU が載っていない PC (AMD Ryzen AI (ベンチマークレス), 先代) では動かない。**

### NPU ドライバが立っているか確認
```bash
lsmod | grep -iE 'amdxdna|amd_pmf|amdxcp|amd_atl'      # すべて表示される
cat /sys/bus/pci/drivers/amdxdna/*/fw_version          # NPU fw 版番
ls -la /dev/accel/accel0                               # accel クラス (new XRT)
```
`amdxdna-dkms` が無い・モジュールが立っていない場合はカーネルドライバを入れます:
```bash
sudo apt update && sudo apt install -y amdxdna-dkms
# (resolute 系カスタムカーネルなら、そのディストロの .deb を使用)
```

---

## 1. XRT (Xilinx Runtime) のインストール

open-xdna は XRT 上に NPU へ GEMM をオフロードする。
Ubuntu のパッケージで入る (amdxdna 版):
```bash
sudo apt install -y xrt-base xrt-base-dev libxrt-npu2 libxrt-utils
```
- `/opt/xilinx/xrt/setup.sh` がインストールされる。
- これだけで `libxrt-npu2` (NPU runtime) が入る。

### shim の起動時の環境 (setup.sh を source する)
shim は起動時に `source /opt/xilinx/xrt/setup.sh` して、以下の環境変数を設定する:
```sh
export XILINX_XRT=/opt/xilinx/xrt
export LD_LIBRARY_PATH=$XILINX_XRT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$XILINX_XRT/python${PYTHONPATH:+:$PYTHONPATH}
```
`source` がインタラクティブ/bash-completion で **hang することがあるので**、
手動で設定する方法 (§4) も用意されている。

---

## 2. llvm-21 (AIE コンパイラ / JIT エンジン)

shim engine は JIT コンパイルに `llvm-21` が必要。`llvm-objcopy` も PATH 上にあること。
```bash
sudo apt install -y llvm-21 2>/dev/null || # 配布元 (resolute / AMD) の llvm-21 を入れる
ls -d /usr/lib/llvm-21
/usr/lib/llvm-21/bin/llvm-objcopy --version  # 確認
```

---

## 3. open-xdna の clone と venv (shimenv / Python 3.14)

```bash
cd ~
git clone https://github.com/Scottcjn/open-xdna.git
cd open-xdna

# Python 3.14 環境 (numpy + ml_dtypes + torch + mlir_aie + llvm-aie 込み)
python3.14 -m venv shimenv
./shimenv/bin/pip install --upgrade pip wheel
```

### MLIR-AIE / AIE / Peano(llvm-aie) を shimenv に入れる
```bash
./shimenv/bin/pip install mlir_aie \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4
./shimenv/bin/pip install llvm-aie \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```
- `mlir_aie` 1.3.4, `llvm-aie` 22.0.0 が入る。
- `aie` モジュールは `mlir_aie` 同梱の `aie.pth` で PATH 加算される
  (例: `shimenv/lib/.../mlir_aie/python/aie/__init__.py`)。

### 依存 (torch / transformers は pip が補う)
shim 起動時に必要になる: torch (CPU), transformers 等。
```bash
./shimenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./shimenv/bin/pip install transformers numpy ml_dtypes py-spy
```
※ 実測の shimenv: torch 2.13.0+cpu, transformers 5.15.1, ml_dtypes 0.6.0

---

## 4. 環境変数の設定 (`source` しない代替案)

`source /opt/xilinx/xrt/setup.sh` が hang する場合、直接設定する:
```bash
export XILINX_XRT=/opt/xilinx/xrt
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH
export PATH=/usr/lib/llvm-21/bin:$PATH

# shimenv アクティベート
cd /home/dino/open-xdna
source ./shimenv/bin/activate
```

### NPU が動くか先にチェック (import チェーン)
```bash
./shimenv/bin/python -c "import aie, aie.iron; print('aie.iron OK')"
./shimenv/bin/python -c "import sys; sys.path.insert(0,'/home/dino/open-xdna/server'); import npu_gemm; print('available=', npu_gemm.available())"
# → available()= True なら OK (False なら LD_LIBRARY_PATH / PYTHONPATH / XILINX_XRT を確認)
```

---

## 5. モデル (HuggingFace)

`server/models.json` の既定:
- `xdna1-mock` — NPU 計算を伴わない配線検証用
- `qwen2.5-0.5b` — 実 NPU GEMM (`Qwen/Qwen2.5-0.5B-Instruct`)

初回は shim 起動時に HuggingFace から自動DL (`~/.cache/huggingface/hub/`)。
別PCで预先 DL しておくか、shim 初回リクエスト時に DL される (JIT と併せて数分待たれる)。

```bash
# 预先 DL (任意)。shim 側に HF 経由で落とす手動DLは不要。
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/.cache/huggingface/hub
```

---

## 6. shim の起動

```bash
cd /home/dino/open-xdna
export XILINX_XRT=/opt/xilinx/xrt
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH
export PATH=/usr/lib/llvm-21/bin:$PATH
source ./shimenv/bin/activate

# 背景启动 (ロギング付き)
nohup python server/xdna_openai_server.py > server/xdna.log 2>&1 &
```

- 初回は全 GEMM / attention shape を JIT コンパイル (`XDNA_NPU_WARMUP=1` デフォルト)。
- 起動が止まっている見栄えになったら `XDNA_NPU_WARMUP=0` で、初回リクエスト時に JIT。

### 検証 (mock で配線チェック)
```bash
curl -s -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"xdna1-mock","messages":[{"role":"user","content":"hi"}]}'
```

---

## 7. Lemonade に登録 (任意)

```bash
export LEMONADE_XDNA_API_KEY='***'   # 空でなくても良い
lemonade cloud install xdna \
  --base-url http://127.0.0.1:8901/v1 \
  --allow-insecure-http
lemonade list    # → xdna.qwen2.5-0.5b  Yes  cloud を確認
```

### Lemonade 経由で実 NPU を使う (port 13305)
```bash
curl -s -X POST http://127.0.0.1:13305/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"xdna.qwen2.5-0.5b","messages":[
        {"role":"user","content":"Say hi in 3 words."}],
       "stream":false}'
```
`npu_busy_percent` / `npu_run_seconds` が伸びれば NPU が動いた証拠。

---

## 8. 重要注意 (このハードでの実測)

### 速度
- **このマシンでは LLM decode は CPU (llama.cpp) が最速 (~35 tok/s)**。
- NPU GEMM オンで decode すると 0.14 tok/s (**250倍遅い**)。
- 理由: XDNA1 は1ランあたり overhead ~50ms だが、decode GEMM (M=1 matvec) は µs級。
  NPU が恩恵を出するのは **大きい M** (長い prefill / バッチ) / CNN / vision のみ。

### fused attention は OFF がデフォルト (2026-08-31確定)
- `XDNA_NPU_ATTENTION_IMPL=fused` (単一dispatch QK^T+softmax+AV) は、
  実 Qwen2.5-0.5B で全24層で分岐 (prefill overflow 残留、NPU bf16 GEMM 量子化)。
- 既定は `perhead` (A/B 検証済み) もしくは `XDNA_NPU_ATTENTION=0` (torch SDPA)。
- 高速化目的で fused をオンにしないこと。

### 単一 NPU コンテキスト
- shim が NPU コンテキストを保持中は、別プロセスは
  `DRM_IOCTL_AMDXDNA_CREATE_HWCTX failed (err=-22)` で **静かに CPU にフォールバック**。
- `npu_gemm.available()` は True のままなので誤認注意。**NPU 利用者は1プロセスまで**。

### 起動失敗 (init/非シェル経由)
- init (PID 1) 経由で始まった shim は `/health` に Empty reply を返すことがあっても
  LISTEN はしている。`server/xdna.log` を確認して再起動。

---

## 9. トラブルシューティング

| 症状 | 確認 / 対処 |
|---|---|
| `available()` が False | `LD_LIBRARY_PATH` / `PYTHONPATH` / `XILINX_XRT` 再設定; `libxrt-npu2` インストール確認 |
| shim が起動して応答しない | `XDNA_NPU_WARMUP=0` にして初回リクエストでJIT; `server/xdna.log` 確認 |
| `aie.iron` import 失敗 | PYTHONPATH に `/opt/xilinx/xrt/python` が含まれる確認 |
| jit が遅い (初回) | 通常数分。`warmup_fused_shapes` で起動時に pre-JIT |
| mock は動く・実モデルが diverge | fused attention の既知問題 → `XDNA_NPU_ATTENTION=0` (CPU) で運用 |

---

## 10. 既知のパス (この環境基準)

- XRT: `/opt/xilinx/xrt/setup.sh`
- llvm-21: `/usr/lib/llvm-21/bin` (llvm-objcopy)
- shimenv: `/home/dino/open-xdna/shimenv` (Python 3.14)
- model cache: `~/.cache/huggingface/hub/`
- shim log: `server/xdna.log`

> 注: `scripts/setup_iron.sh` (upstream) は Python 3.13 前提・`cd /root/mlir-aie` 等
> パスが破損しているため本手順では **使用しない**。shimenv に直接 aie/llvm-aie を
> インストールする方法が正攻法。
