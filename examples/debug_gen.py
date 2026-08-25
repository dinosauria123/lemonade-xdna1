import sys; sys.path.insert(0, 'server')
import engine, traceback, torch
e = engine.XDna1LLM('Qwen/Qwen2.5-0.5B-Instruct', True)
text = e.tokenizer.apply_chat_template(
    [{'role':'user','content':'What is 2+2?'}], tokenize=False, add_generation_prompt=True)
input_ids = e.tokenizer(text, return_tensors="pt").input_ids
print("prompt_len", input_ids.shape[1], flush=True)
with engine._engine_lock:
    e.model.to(e.device)
    past = None
    cur = input_ids
    out = e.model(cur, past_key_values=past, use_cache=True)
    print("logits shape", out.logits.shape, flush=True)
    logits = out.logits[0, -1]
    print("last logits shape", logits.shape, "isnan", bool(torch.isnan(logits).any()), flush=True)
    tok = e._sample(logits, 0.0, 1)
    print("tok", tok, "eos", e.tokenizer.eos_token_id, flush=True)
