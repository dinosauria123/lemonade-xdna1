#!/usr/bin/env python3
"""Benchmark xdna shim: time a fixed prompt -> N completion tokens.
Usage: python bench_xdna.py [port] [max_tokens] [label]
"""
import json, time, urllib.request, sys

port = sys.argv[1] if len(sys.argv) > 1 else "8901"
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 64
label = sys.argv[3] if len(sys.argv) > 3 else ""

url = f"http://127.0.0.1:{port}/v1/chat/completions"
body = json.dumps({
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Write a short story about a robot gardener."}],
    "max_tokens": max_tokens,
    "temperature": 0.0,
    "stream": False,
}).encode()

# warm request (engine hot, kernels cached)
for i in range(2):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    dt = time.time() - t0
    n = d["usage"]["completion_tokens"]
    print(f"warmup{i+1}: {n} tok in {dt:.2f}s")

# timed request
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.load(r)
dt = time.time() - t0
n = d["usage"]["completion_tokens"]
print(f"[{label}] {n} tok in {dt:.2f}s  =  {n/dt:.2f} tok/s")
print(f"[{label}] text: {d['choices'][0]['message']['content'][:200]!r}")
