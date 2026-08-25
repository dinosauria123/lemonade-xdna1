#!/usr/bin/env python3
"""End-to-end: NPU attention path must produce correct output.
Compares generation text with and without XDNA_NPU_ATTENTION."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import importlib
import engine as engine_mod

PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]

def run(enable_attn):
    os.environ["XDNA_NPU_ATTENTION"] = "1" if enable_attn else "0"
    importlib.reload(engine_mod)
    e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
    text = "".join(e.generate(PROMPT, {"max_tokens": 24, "temperature": 0.0, "top_k": 1}))
    return text

t_off = run(False)
print("CPU-attention :", repr(t_off[:80]))
t_on = run(True)
print("NPU-attention :", repr(t_on[:80]))
# token-level agreement (temperature 0 -> greedy; should match if correct)
agree = t_off.strip() == t_on.strip()
print("EXACT MATCH:", agree)
print("RESULT:", "ATTENTION PATH CORRECT" if agree else "DIVERGED -- check attention wrapper")
