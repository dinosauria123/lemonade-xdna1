#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# open-xdna :: small real-LLM engine for the XDNA1 shim.
#
# Base engine: Qwen2.5-0.5B-Instruct in bf16 on CPU (torch), manual decode loop
# with KV cache and temperature/top-k sampling. This is a REAL model — the text
# it produces is genuine model output (not a mock).
#
# NPU integration: when the XDNA1 NPU is available and offload is enabled, the
# per-layer linear GEMMs (q/k/v/o/gate/up/down proj) are dispatched to the NPU
# via the IRON single_core bf16 matmul kernel (npu_gemm.py).
#
# Default: EVERY GEMM is offloaded (prefill and decode). Decode at M=1 pads to
# the 32-row tile, so the NPU computes 32x more than strictly needed — that is
# the cost of "NPU always in the loop". Set XDNA_NPU_GEMM_PREFILL_ONLY=1 to
# offload only M>=32 GEMMs (faster, NPU idle during short decode).
#
# Every NPU-dispatched GEMM is counted in npu_gemm.stats, so the API surface
# can report how much of a request actually ran on the NPU.

import os
import threading

import numpy as np

import npu_gemm

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_TOKENS = 512

_engine_lock = threading.Lock()
_engine = None


def _offload_enabled():
    return os.environ.get("XDNA_NPU_GEMM", "1") not in ("0", "false", "no")


def _npu_attention_enabled():
    """Route the attention matmuls (QK^T, AV) through the NPU (default ON).

    Verified A/B: with the causal-mask fix (causality from S_q==S_k geometry,
    not from the cache object), greedy decode matches CPU attention exactly
    (examples/test_engine_attention.py, EXACT MATCH). Set
    XDNA_NPU_ATTENTION=0 to fall back to torch SDPA.
    """
    return os.environ.get("XDNA_NPU_ATTENTION", "1") in ("1", "true", "yes")


def _prefill_only():
    return os.environ.get("XDNA_NPU_GEMM_PREFILL_ONLY", "0") in ("1", "true", "yes")


class XDna1LLM:
    def __init__(self, model_id: str, npu_gemm_enabled: bool = True):
        import torch  # imported lazily so the HTTP server starts without it

        self.torch = torch
        self.model_id = model_id
        self.npu_gemm_enabled = npu_gemm_enabled
        self.bf16 = torch.bfloat16
        self.device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.bf16, low_cpu_mem_usage=True
        ).eval()

        self.layers = self.model.model.layers
        self.npu_linear_count = 0
        if npu_gemm_enabled:
            for layer in self.layers:
                for lin in (
                    layer.self_attn.q_proj, layer.self_attn.k_proj,
                    layer.self_attn.v_proj, layer.self_attn.o_proj,
                    layer.mlp.gate_proj, layer.mlp.up_proj,
                    layer.mlp.down_proj,
                ):
                    self._wrap_linear(lin)
            if _npu_attention_enabled():
                self._install_npu_attention()

        # NPU status snapshot (for /health and diagnostics)
        self.npu_available = npu_gemm.available()
        self.npu_error = "" if self.npu_available else npu_gemm.error()

    # -- NPU attention override -------------------------------------------------

    def _install_npu_attention(self):
        """Replace each layer's SDPA with the NPU attention core.

        The Q/K/V/O projections stay NPU-GEMM-wrapped (see _wrap_linear). We
        keep transformers' RoPE + KV-cache handling and only swap the
        scaled-dot-product attention math for npu_gemm.npu_attention_core, which
        routes the QK^T and AV matmuls through the same NPU kernel.
        """
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        for layer in self.layers:
            attn = layer.self_attn
            attn._npu_forward_orig = attn.forward

            def make_forward(attn):
                cfg = attn.config
                n_heads_q = cfg.num_attention_heads
                n_heads_kv = cfg.num_key_value_heads
                n_rep = n_heads_q // n_heads_kv
                head_dim = attn.head_dim
                scale = attn.scaling

                def forward(hidden_states, position_embeddings, attention_mask,
                            past_key_values=None, **kwargs):
                    import torch  # already loaded in __init__
                    import numpy as np
                    from ml_dtypes import bfloat16
                    input_shape = hidden_states.shape[:-1]  # (1, S_q) for batch=1
                    hidden_shape = (*input_shape, -1, head_dim)
                    q = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    k = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    if past_key_values is not None:
                        k, v = past_key_values.update(k, v, attn.layer_idx)
                    # batch is 1; to numpy bf16 [n_heads, S, D]
                    Q = q.detach().to(torch.float32).numpy()[0].astype(bfloat16)
                    K = k.detach().to(torch.float32).numpy()[0].astype(bfloat16)
                    V = v.detach().to(torch.float32).numpy()[0].astype(bfloat16)
                    out_np = npu_gemm.npu_attention_core(
                        Q, K, V, scale, n_rep,
                        # Causality from geometry, not from the cache object:
                        # with use_cache=True the model hands us an (empty)
                        # DynamicCache even for the prefill, so
                        # `past_key_values is None` is NOT a reliable prefill
                        # signal. Prefill: S_q == S_k -> apply causal mask.
                        # Decode: S_q(=1) < S_k -> query sits at the end, no
                        # mask needed.
                        causal=(Q.shape[1] == K.shape[1]))
                    # npu_attention_core returns [n_heads, S_q, D] (batch sliced off).
                    # Re-add the (1,) leading dim to match transformers' Qwen2Attention,
                    # which reshapes attn_output (batch, n_heads, S_q, D) via
                    # .transpose(1, 2).reshape(batch, S_q, n_heads*D). With the extra
                    # dim this is: out.view(1, n_heads, S_q, D).transpose(1, 2).reshape
                    # (*input_shape, -1); the previous .transpose(1, 2) on the raw
                    # 3-D tensor swapped head<->seq and produced garbage.
                    out_t = torch.from_numpy(out_np).to(attn.q_proj.weight.dtype)
                    attn_output = out_t.view(
                        1, n_heads_q, input_shape[1], head_dim
                    ).transpose(1, 2).reshape(
                        *input_shape, -1).contiguous()
                    attn_output = attn.o_proj(attn_output)
                    return attn_output, None

                return forward

            attn.forward = make_forward(attn)

    # -- NPU GEMM wrapper ----------------------------------------------------

    def _wrap_linear(self, lin):
        """Route this Linear's GEMM through the NPU when profitable/available."""
        torch = self.torch
        from ml_dtypes import bfloat16 as _bf16

        # torch 2.x .numpy() rejects bf16 tensors, so the conversion path is
        # bf16 -> f32 tensor -> f32 numpy -> astype(bfloat16) (exact, fast).
        # Weights are constant during generation: convert once, here.
        try:
            lin.npu_weight_np = (
                lin.weight.detach().to(torch.float32).contiguous()
                .numpy().astype(_bf16).copy())
        except Exception:  # noqa: BLE001 - weight conversion failed
            lin.npu_weight_np = None
        lin.npu_gemm_enabled = bool(
            self.npu_gemm_enabled and lin.npu_weight_np is not None)

        def forward(x):
            if not lin.npu_gemm_enabled:
                return lin.forward_orig(x)
            if x.dim() < 2:
                return lin.forward_orig(x)
            # Linear consumes the last dim; the GEMM "M" is all leading dims
            # (transformers feeds [batch, seq, hidden] — 3D — into the linears).
            M = 1
            for d in x.shape[:-1]:
                M *= d
            if M and (M >= 32 or not _prefill_only()) and npu_gemm.available():
                try:
                    a = (x.detach().to(torch.float32).reshape(M, -1)
                         .contiguous().numpy().astype(_bf16))
                    out = npu_gemm.matmul_bf16(a, lin.npu_weight_np)  # f32 [M, N]
                    t = torch.from_numpy(out).to(self.bf16).reshape(
                        *x.shape[:-1], -1)
                    if lin.bias is not None:
                        t = t + lin.bias.to(self.bf16)
                    return t
                except Exception as e:  # noqa: BLE001 - fall back, log once
                    if not getattr(lin, "_npu_warned", False):
                        lin._npu_warned = True
                        import sys
                        print(f"[engine] NPU GEMM fallback "
                              f"({lin.__class__.__name__}): {type(e).__name__}: {e}",
                              file=sys.stderr, flush=True)
            return lin.forward_orig(x)

        lin.forward_orig = lin.forward
        lin.forward = forward
        self.npu_linear_count += 1

    # -- generation ------------------------------------------------------------

    def _sample(self, logits, temperature, top_k):
        torch = self.torch
        if temperature is None or temperature <= 0:
            return int(torch.argmax(logits))
        z = logits.float() / temperature
        if top_k and top_k > 0:
            k = min(top_k, z.shape[-1])
            cut = torch.topk(z, k).values[-1]
            z = torch.where(z < cut, torch.full_like(z, float("-inf")), z)
        p = torch.softmax(z, dim=-1)
        return int(torch.multinomial(p, 1).item())

    def generate(self, messages, params):
        """Yield text deltas for one chat completion."""
        torch = self.torch
        max_tokens = int(params.get("max_tokens") or DEFAULT_MAX_TOKENS)
        temperature = params.get("temperature", 1.0)
        top_k = int(params.get("top_k") or 20)

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        prompt_len = input_ids.shape[1]

        stats_before = dict(npu_gemm.stats)

        with _engine_lock:
            self.model.to(self.device)
            past = None
            cur = input_ids
            out_tokens = []
            generated = 0
            with torch.no_grad():
                while generated < max_tokens:
                    out = self.model(cur, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[0, -1]
                    # stop on EOS
                    if logits.shape[-1] > 0:
                        tok = self._sample(logits, temperature, top_k)
                    else:
                        break
                    out_tokens.append(tok)
                    generated += 1
                    if tok == self.tokenizer.eos_token_id:
                        break
                    cur = self.torch.tensor([[tok]], dtype=torch.int32)
                    delta = self.tokenizer.decode(
                        [tok], skip_special_tokens=True, spaces_between_special_tokens=False
                    )
                    if delta:
                        yield delta

        # NPU accounting for this request
        s = npu_gemm.stats
        self.last_npu_gemm_count = s["gemms"] - stats_before["gemms"]
        self.last_prompt_tokens = prompt_len

    def npu_report(self):
        s = npu_gemm.stats
        return {
            "npu_available": self.npu_available,
            "npu_error": self.npu_error,
            "gemm_offload": self.npu_gemm_enabled,
            "npu_linears_wrapped": self.npu_linear_count,
            "total_npu_gemms": s["gemms"],
            "total_npu_flops_gflop": s["flops"] / 1e9,
            "jit_compiles": s["compile_calls"],
            "npu_run_seconds": round(s["run_s"], 3),
            "last_request_npu_gemms": getattr(self, "last_npu_gemm_count", 0),
        }


def get_engine(model_id: str | None = None) -> XDna1LLM:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = XDna1LLM(model_id or DEFAULT_MODEL, _offload_enabled())
        return _engine
