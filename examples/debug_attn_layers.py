#!/usr/bin/env python3
"""Debug: compare NPU-attention forward vs original SDPA forward, layer by layer.

Runs one prefill with the engine's NPU attention installed, capturing each
layer's input, then re-runs ONLY the attention block (original vs NPU) on that
captured input and compares outputs. Also compares final prefill logits with
NPU attention on vs off.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import numpy as np
import torch

PROMPT = [{"role": "user", "content": "What is 2+2? Answer in one word."}]

# ---------- NPU attention ON ----------
os.environ["XDNA_NPU_ATTENTION"] = "1"
import engine as engine_mod
e = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
print("attention installed on layer0:", hasattr(e.layers[0].self_attn, "forward")
      and "npu_attention_core" in e.layers[0].self_attn.forward.__code__.co_names or
      "_npu_forward_orig" in dir(e.layers[0].self_attn))

text = e.tokenizer.apply_chat_template(PROMPT, tokenize=False, add_generation_prompt=True)
input_ids = e.tokenizer(text, return_tensors="pt").input_ids
S = input_ids.shape[1]
print("prompt_len", S)

model = e.model
model.eval()

# ---- capture layer inputs for one prefill (no past) ----
with torch.no_grad():
    captured = []
    orig_forwards = {}
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig = attn._npu_forward_orig
        npu_fwd = attn.forward  # the NPU wrapper installed by the engine

        def hook(hidden_states, position_embeddings, attention_mask,
                 past_key_values=None, **kw):
            hs, pe, am = hidden_states, position_embeddings, attention_mask
            captured.append((hs.clone(), pe, am is not None))
            # run the ORIGINAL forward (it does NOT touch cache when past is None)
            out_o, _ = orig(hs, pe, am)
            out_n, _ = npu_fwd(hs, pe, am)
            d = (out_o.float() - out_n.float()).abs()
            cos = float((out_o.float().flatten() @ out_n.float().flatten()) /
                        (out_o.float().flatten().norm() * out_n.float().flatten().norm() + 1e-9))
            captured[-1] = (hs.clone(), pe, am is not None, cos, d.max().item(), d.mean().item())
            return out_n, None

        attn.forward = hook
    out = model(input_ids, use_cache=False)
    logits_npu = out.logits[0, -1].float().clone()
    print("prefill done (NPU attention), logits nan:", bool(torch.isnan(logits_npu).any()))
    top5 = torch.topk(logits_npu, 5)
    print("top5:", [(t, e.tokenizer.decode([t])) for t in top5.indices.tolist()])
    for i, c in enumerate(captured):
        hs, pe, had_mask, cos, dmax, dmean = c
        print(f"layer {i:2d}: had_mask={had_mask} cos={cos:.6f} dmax={dmax:.4f} dmean={dmean:.6f}")

# ---------- NPU attention OFF: full prefill logits for comparison ----------
os.environ["XDNA_NPU_ATTENTION"] = "0"
import importlib
importlib.reload(engine_mod)
e2 = engine_mod.XDna1LLM("Qwen/Qwen2.5-0.5B-Instruct", True)
with torch.no_grad():
    out2 = e2.model(input_ids, use_cache=False)
logits_cpu = out2.logits[0, -1].float()
d = (logits_npu - logits_cpu).abs()
print(f"FINAL LOGITS: dmax={d.max().item():.4f} dmean={d.mean().item():.6f}")
print("argmax equal:", int(logits_npu.argmax()) == int(logits_cpu.argmax()),
      "(npu:", e.tokenizer.decode([int(logits_npu.argmax())]), "cpu:", e.tokenizer.decode([int(logits_cpu.argmax())]), ")")
print("done")
