#!/usr/bin/env python3
"""Decompose overflow residual: is it bf16 QK^T quant amplified by softmax?
If yes, it is answerable by raising score-quant precision (fp32-accumulate QK^T,
only quantize the AV weight, or use a bf16->fp16 score accumulator)."""
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

def unpacked_ref(q,k,v):
    out=np.zeros((n_hq,S_q,D),np.float32)
    for g in range(n_hkv):
        for h in range(n_rep):
            qq=q[g*n_rep+h]*scale; T=qq@k[g].T
            T=np.where(i>=j,T,-np.inf); T=T-T.max(1,keepdims=True)
            P=np.exp(T); P/=P.sum(1,keepdims=True); out[g*n_rep+h]=P@v[g]
    return out

q,k,v=mk(80.0)
A=unpacked_ref(q,k,v)

# (1) bf16 QK^T quant error vs fp32
maxqk=0
for g in range(n_hkv):
    for h in range(n_rep):
        r=(q*scale)[g*n_rep+h]@k[g].T
        rn=npu_gemm.matmul_bf16((q*scale)[g*n_rep+h].astype(bfloat16),k[g].astype(bfloat16))
        maxqk=max(maxqk,np.abs(rn-r).max()/(np.abs(r).max()+1e-9))
print(f"QK^T bf16 max rel err (|T|~{np.abs(q*scale).max():.0f}): {maxqk:.4f}")

# (2) isolate: bf16 QK^T -> exact softmax -> exact AV (NO AV quant, NPU not needed)
i=np.arange(S_q)[:,None]; j=np.arange(S_k)[None,:]
isoret=np.zeros((n_hq,S_q,D),np.float32)
for g in range(n_hkv):
    for h in range(n_rep):
        rn=npu_gemm.matmul_bf16((q*scale)[g*n_rep+h].astype(bfloat16),k[g].astype(bfloat16))
        Tm=np.where(i>=j,rn,-np.inf); Tm=Tm-Tm.max(1,keepdims=True)
        P=np.exp(Tm); P/=P.sum(1,keepdims=True); isoret[g*n_rep+h]=P@v[g]
rel2=np.abs(isoret-A).max()/(np.abs(A).max()+1e-9)
print(f"(2) bf16-QK^T only (exact softmax, exact AV): max rel = {rel2:.4f}")

# (3) exact QK^T -> bf16 softmax -> bf16 AV on NPU (isolates AV quant contribution)
avret=np.zeros((n_hq,S_q,D),np.float32)
for g in range(n_hkv):
    for h in range(n_rep):
        T=(q*scale)[g*n_rep+h]@k[g].T
        Tm=np.where(i>=j,T,-np.inf); Tm=Tm-Tm.max(1,keepdims=True)
        P=np.exp(Tm); P/=P.sum(1,keepdims=True)
        ov=npu_gemm.matmul_bf16(P.astype(bfloat16), v[g].astype(bfloat16).T)
        avret[g*n_rep+h]=ov[:S_q,:D]
rel3=np.abs(avret-A).max()/(np.abs(A).max()+1e-9)
print(f"(3) exact-QK^T then bf16-AV on NPU: max rel = {rel3:.4f}")

print()
if abs(rel2-0.77)<0.10:
    print(">>> CONFIRMED: residual = bf16 QK^T mantissa quant amplified by "
          "softmax. It IS answerable: raise score-quant precision.")
elif rel3 < rel2:
    print(">>> residual is AV-quant dominated (softmax of fp32 T into bf16 AV).")
else:
    print(">>> residual split between QK^T and AV quantization.")
