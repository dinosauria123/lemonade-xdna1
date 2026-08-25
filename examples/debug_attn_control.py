#!/usr/bin/env python3
"""Control experiment: same engine instance, attention ON vs OFF, use_cache=False.

Isolates whether the NPU attention path (even with causal=True, past=None)
moves the final prefill logits, vs NPU-GEMM nondeterminism between runs.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import torch

os.environ["XDNA_NPU_ATTENTION"] = "1"
import engine as engine_mod
e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
text = e.tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    tokenize=False, add_generation_prompt=True)
input_ids = e.tokenizer(text, return_tensors="pt").input_ids
print("prompt_len", input_ids.shape[1])
model = e.model
model.eval()

def prefill():
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
    return out.logits[0, -1].float().clone()

def cmp(name, a, b):
    d = (a - b).abs()
    ia, ib = int(a.argmax()), int(b.argmax())
    print(f"{name}: dmax={d.max().item():.4f} dmean={d.mean().item():.6f} "
          f"argmax {e.tokenizer.decode([ia])!r} vs {e.tokenizer.decode([ib])!r} equal={ia==ib}")

# 1) attention ON (installed at init)
L_on1 = prefill()
L_on2 = prefill()  # determinism check, same config
cmp("ON  vs ON(dup)  ", L_on1, L_on2)

# 2) switch all layers back to the original SDPA forward
for layer in model.model.layers:
    layer.self_attn.forward = layer.self_attn._npu_forward_orig
L_off1 = prefill()
L_off2 = prefill()
cmp("OFF vs OFF(dup)  ", L_off1, L_off2)
cmp("ON  vs OFF       ", L_on1, L_off1)
print("done")
