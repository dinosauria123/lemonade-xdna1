#!/usr/bin/env python3
"""End-to-end: NPU attention path must produce correct output.

Three-way comparison at greedy (temperature 0):
  cpu     -- XDNA_NPU_ATTENTION=0 (torch SDPA, baseline)
  perhead -- XDNA_NPU_ATTENTION=1, IMPL=perhead (A/B exact-verified)
  fused   -- XDNA_NPU_ATTENTION=1, IMPL=fused (M3 single-dispatch, default)

Reports token-exact match vs the CPU baseline and wall time for each.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import importlib
import engine as engine_mod

PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
PARAMS = {"max_tokens": 24, "temperature": 0.0, "top_k": 1}


def run(attn, impl):
    os.environ["XDNA_NPU_ATTENTION"] = "1" if attn else "0"
    if attn:
        os.environ["XDNA_NPU_ATTENTION_IMPL"] = impl
    else:
        os.environ.pop("XDNA_NPU_ATTENTION_IMPL", None)
    importlib.reload(engine_mod)
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    t0 = time.perf_counter()
    text = "".join(e.generate(PROMPT, PARAMS))
    dt = time.perf_counter() - t0
    return text, dt


base, dt_cpu = run(False, None)
print(f"cpu     : {base.strip()!r}  ({dt_cpu:.2f}s)")

per, dt_per = run(True, "perhead")
print(f"perhead : {per.strip()!r}  ({dt_per:.2f}s)  exact={per.strip() == base.strip()}")

fus, dt_fus = run(True, "fused")
print(f"fused   : {fus.strip()!r}  ({dt_fus:.2f}s)  exact={fus.strip() == base.strip()}")

agree = base.strip() == fus.strip() == per.strip()
print("\nALL THREE AGREE:", agree)
if not agree:
    print("CPU   :", repr(base))
    print("PERHD :", repr(per))
    print("FUSED :", repr(fus))
print("RESULT:", "ATTENTION PATH CORRECT" if agree else "DIVERGED -- check attention wrapper")
print(f"\nSPEEDUP (cpu vs fused): {dt_cpu / max(dt_fus, 1e-9):.1f}x")
