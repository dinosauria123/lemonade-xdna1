#!/usr/bin/env python3
"""DECISIVE: which half causes the overflow residual?

Current path: bf16 QK^T (NPU) -> fp32 scores -> exact softmax -> bf16 AV (NPU).
Test 4 variants vs exact fp32 ground truth + Option-1 (i16 AV):
  (a) current: bf16 QK^T + bf16 AV
  (b) FIX-QK : fp32 QK^T + bf16 AV
  (c) FIX-AV : bf16 QK^T + fp32 AV
  (d) FULL   : fp32 QK^T + fp32 AV
  (e) OPT-1  : bf16 QK^T + i16 fixed-point AV
Whichever single change drops the residual to ~0 is THE culprit.
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
    return P@v

def av_i16(P,v):
    OUT=np.zeros((n_hq,S_q,D),np.float32); SF=4096
    for g in range(n_hkv):
        for h in range(n_rep):
            idx=g*n_rep+h
            Pi16=(P[idx].astype(np.float32)*SF).round().astype(np.int16)
            rr=npu_gemm.matmul_bf16(Pi16.astype(bfloat16), v[g].astype(bfloat16).T)
            OUT[idx]=rr[:S_q,:D]/SF
    return OUT

def full(res,G):
    return np.abs(res-G).max()/(np.abs(G).max()+1e-9)

q,k,v=mk(80.0)
G=exact(q,k,v)
Tn=qk_npu(q,k); Th=qk_host(q,k)
Pn=sm(Tn); Ph=sm(Th)
av16=av_i16(Pn,v)

print(f"|T|~{np.abs(q*scale).max():.0f}  |exact|max={np.abs(G).max():.3f}")
print(f"(a) current  bf16-QK^T + bf16-AV   : {full(av_bf16(Pn,v),G):.4f}")
print(f"(b) FIX-QK   fp32-QK^T + bf16-AV   : {full(av_bf16(Ph,v),G):.4f}")
print(f"(c) FIX-AV   bf16-QK^T + fp32-AV   : {full(av_host(Pn,v),G):.4f}")
print(f"(d) FULL     fp32-QK^T + fp32-AV   : {full(av_host(Ph,v),G):.4f}")
print(f"(e) OPT-1    bf16-QK^T + i16-AV    : {full(av16,G):.4f}")
