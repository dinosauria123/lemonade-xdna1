#!/usr/bin/env python3
"""Root cause of overflow residual + Option-1 (i16 AV) check, with robust NaN
handling and an NPU-state-integrity probe.

We must rule out that matmul_bf16 is spitting NaN because the single_core shared
result buffer is contaminated by the large-magnitude QK^T matmuls. Probe: after
100+ large GEMMs, run a fresh tiny bounded bf16 matmul; if THAT is NaN, the NPU
device is in a corrupt state and every residual number is meaningless until we
reset (fresh kernel / fresh device) between QK^T and AV.
"""
import os, sys
sys.path.insert(0, "/home/dino/open-xdna/server")
os.environ["XDNA_NPU_ATTENTION_IMPL"] = "fused"; os.environ["XDNA_NPU_ATTENTION"]="1"
import numpy as np
from ml_dtypes import bfloat16
import npu_gemm

D=64; n_rep=7; n_hkv=2; n_hq=n_rep*n_hkv; S_q,S_k=41,64
scale=D**-0.5; i=np.arange(S_q)[:,None]; j=np.arange(S_k)[None,:]

def mk(sigma,seed=3):
    rng=np.random.default_rng(seed)
    q=rng.standard_normal((n_hq,S_q,D)).astype(np.float32)*sigma
    k=rng.standard_normal((n_hkv,S_k,D)).astype(np.float32)*sigma*0.1
    v=rng.standard_normal((n_hkv,S_k,D)).astype(np.float32)
    return q,k,v

def exact(q,k,v):
    out=np.zeros((n_hq,S_q,D),np.float32)
    for g in range(n_hkv):
        for h in range(n_rep):
            T=(q*scale)[g*n_rep+h]@k[g].T
            T=np.where(i>=j,T,-np.inf); T=T-T.max(1,keepdims=True)
            P=np.exp(T); P/=P.sum(1,keepdims=True); out[g*n_rep+h]=P@v[g]
    return out

def qk_npu(q,k):
    out=np.zeros((n_hq,S_q,S_k),np.float32)
    for g in range(n_hkv):
        for h in range(n_rep):
            out[g*n_rep+h]=npu_gemm.matmul_bf16((q*scale)[g*n_rep+h].astype(bfloat16),k[g].astype(bfloat16))
    return out

def qk_host(q,k):
    out=np.zeros((n_hq,S_q,S_k),np.float32)
    for g in range(n_hkv):
        for h in range(n_rep):
            out[g*n_rep+h]=(q*scale)[g*n_rep+h]@k[g].T
    return out

def sm(T):
    T=np.where(i>=j,T,-np.inf); T=T-T.max(1,keepdims=True); P=np.exp(T); P/=P.sum(1,keepdims=True); return P

def av_bf16(P,v):
    out=np.zeros((n_hq,S_q,D),np.float32)
    for g in range(n_hkv):
        for h in range(n_rep):
            idx=g*n_rep+h
            out[idx]=npu_gemm.matmul_bf16(P[idx].astype(bfloat16), v[g].astype(bfloat16).T)[:S_q,:D]
    return out

def av_host(P,v):
    out=np.zeros((n_hq,S_q,D),np.float32)
    for g in range(n_hkv):
        for h in range(n_rep):
            out[g*n_rep+h]=P[g*n_rep+h]@v[g]
    return out

def av_i16(P,v):
    OUT=np.zeros((n_hq,S_q,D),np.float32); SF=4096
    for g in range(n_hkv):
        for h in range(n_rep):
            idx=g*n_rep+h
            Pi16=(P[idx].astype(np.float32)*SF).round().astype(np.int16)
            rr=npu_gemm.matmul_bf16(Pi16.astype(bfloat16), v[g].astype(bfloat16).T)
            OUT[idx]=rr[:S_q,:D]/SF
    return OUT

def rel(res,G):
    Gf=np.nan_to_num(G)
    return np.abs(res-Gf).max()/(np.abs(Gf).max()+1e-9), bool(np.isnan(res).any() or np.isinf(res).any())

q,k,v=mk(80.0)
G=exact(q,k,v)

print("=== NPU STATE INTEGRITY PROBE (run AFTER many large GEMMs) ===")
for _ in range(120):
    _=npu_gemm.matmul_bf16(np.random.randn(64,64).astype(bfloat16), np.random.randn(64,64).astype(bfloat16))
probe_small = np.random.randn(64,64).astype(bfloat16)
probe_w   = np.random.randn(64,64).astype(bfloat16)
probe = npu_gemm.matmul_bf16(probe_small, probe_w)
print(f"fresh tiny GEMM after 120x large: mean={probe.mean():.3e} max={probe.max():.3e} nan={np.isnan(probe).any()}")
if np.isnan(probe).any():
    print(">>> NPU STATE IS CORRUPT -- residual numbers below are UNRELIABLE. Need reset between QK^T and AV.")
    print(">>> Option-1 (i16 AV) MUST reset/clear state first or it inherits the same NaN.")
else:
    print(">>> NPU state healthy. Continuing residual decomposition.")

Tn=qk_npu(q,k); Th=qk_host(q,k)
Pn=sm(Tn); Ph=sm(Th)
print(f"\n|T|~{np.abs(q*scale).max():.0f}  |exact|max={np.abs(G).max():.3f}")

a=av_bf16(Pn,v); ra,na=rel(a,G)
b=av_bf16(Ph,v); rb,nb=rel(b,G)
c=av_host(Pn,v); rc,nc=rel(c,G)
d=av_host(Ph,v); rd,nd=rel(d,G)
e=av_i16(Pn,v); re,ne=rel(e,G)

print(f"(a) current  bf16-QK^T + bf16-AV   : rel={ra:.4f} nan={na}")
print(f"(b) FIX-QK   fp32-QK^T + bf16-AV   : rel={rb:.4f} nan={nb}")
print(f"(c) FIX-AV   bf16-QK^T + fp32-AV   : rel={rc:.4f} nan={nc}")
print(f"(d) FULL     fp32-QK^T + fp32-AV   : rel={rd:.4f} nan={nd}")
print(f"(e) OPT-1    bf16-QK^T + i16-AV    : rel={re:.4f} nan={ne}")
