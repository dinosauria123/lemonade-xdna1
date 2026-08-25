#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# open-xdna :: OpenAI-compatible shim server for wiring XDNA1 inference into AMD
# Lemonade (lemond) via its built-in "cloud" OpenAI-compatible forwarder backend.
#
# Why a shim: Lemonade's native backends (llamacpp, fastflowlm, ...) are
# compile-time registered (LEMON_BACKENDS in CMakeLists.txt) and target XDNA2
# tooling. The "cloud" backend, however, forwards to ANY OpenAI-compatible
# endpoint at runtime (`lemonade cloud install <provider> --base-url ...`).
# This server is the OpenAI-compatible face of the open-xdna runtime: lemond
# registers it as provider "xdna", discovers models from GET /v1/models, and
# forwards /v1/chat/completions to us.
#
# Backends:
#   mock   — deterministic echo model; proves the lemond<->shim wiring end to
#            end with no NPU work involved. Output is clearly labeled.
#   xdna1  — the real engine slot. Not implemented yet (open-xdna currently
#            ships kernels, not a full LLM engine). Registered as a stub that
#            fails loudly rather than faking output.
#
# Run:
#   python3 server/xdna_openai_server.py
# Environment:
#   XDNA_OAI_HOST        bind address           (default 127.0.0.1)
#   XDNA_OAI_PORT        port                   (default 8901)
#   XDNA_OAI_KEY         if set, require "Authorization: Bearer <key>"
#   XDNA_OAI_MODELS      path to models.json    (default: server/models.json)
#
# Register with Lemonade (lemond must be running, default port 13305):
#   export LEMONADE_XDNA_API_KEY=***          # any non-empty value
#   lemonade cloud install xdna \
#     --base-url http://127.0.0.1:8901/v1 \
#     --allow-insecure-http
#   lemonade list | grep xdna
#   lemonade run xdna.xdna1-mock

import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "models": [
        {
            "id": "xdna1-mock",
            "name": "XDNA1 mock (wiring test)",
            "context_length": 4096,
            "backend": "mock",
            "supports_chat": True,
        }
    ]
}


def load_models(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", [])
        if models:
            return models
        print(f"WARNING: {path} has no models, using built-in defaults", file=sys.stderr)
    return DEFAULT_MODELS["models"]


# ---------------------------------------------------------------------------
# Rolling NPU busy % (duty cycle)
#
# The XDNA1 firmware / kernel PMF on this platform (Ryzen 7 8845HS / Hawk
# Point) does NOT expose an NPU utilization sensor via the amdxdna QUERY_SENSORS
# ioctl (amd_pmf_get_npu_data -> -ENODEV), so tools like lemond's
# /system-stats cannot read a hardware NPU %. The shim, however, runs the NPU
# and already accumulates npu_gemm.stats["run_s"] (wall time spent in NPU GEMM
# kernels). A rolling duty cycle 100*Δrun_s/Δt over a short window is a genuine
# measure of NPU busy-ness for the workload that actually uses the silicon here.
# ---------------------------------------------------------------------------

class NpuBusy:
    """Rolling NPU duty cycle derived from npu_gemm.stats["run_s"]."""

    def __init__(self, window=1.0, poll=0.2):
        self.window = float(window)
        self.poll = float(poll)
        self._lock = threading.Lock()
        self._samples = []  # list of (monotonic_t, run_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    @staticmethod
    def _read_run_s():
        try:
            import npu_gemm

            return float(npu_gemm.stats.get("run_s", 0.0))
        except Exception:
            return 0.0

    def _poll(self):
        while not self._stop.is_set():
            time.sleep(self.poll)
            with self._lock:
                self._samples.append((time.perf_counter(), self._read_run_s()))
                cutoff = time.perf_counter() - self.window * 4
                while self._samples and self._samples[0][0] < cutoff:
                    self._samples.pop(0)

    def percent(self):
        with self._lock:
            self._samples.append((time.perf_counter(), self._read_run_s()))
            if len(self._samples) < 2:
                return 0.0
            t0, rs0 = self._samples[0]
            t1, rs1 = self._samples[-1]
            dt = t1 - t0
            if dt <= 0:
                return 0.0
            return max(0.0, min(100.0, 100.0 * (rs1 - rs0) / dt))

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# model backends
#
# A backend yields text deltas (str) for one generation call. It must not
# swallow its own errors: the HTTP layer turns exceptions into 500s.
# ---------------------------------------------------------------------------


class MockBackend:
    """Deterministic echo model for wiring verification.

    Not a language model: it restates conversation facts so a human (or a
    test) can see that the request actually traversed lemond -> shim -> back.
    """

    name = "mock"

    def generate(self, model, messages, params):
        model_id = model["id"]
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    last_user = content
                elif isinstance(content, list):  # OpenAI vision content parts
                    last_user = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                break

        n_msgs = len(messages)
        text = (
            f"[xdna-shim mock model '{model_id}'] wiring OK.\n"
            f"Conversation so far: {n_msgs} message(s). "
            f"Last user message: {last_user!r}. "
            f"Params: temperature={params.get('temperature')}, "
            f"max_tokens={params.get('max_tokens')}. "
            f"This text was produced by the open-xdna shim mock backend "
            f"running on {time.strftime('%Y-%m-%d %H:%M:%S')}."
        )
        max_tokens = params.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            words = text.split(" ")
            clipped = " ".join(words[: max_tokens * 4])  # crude 4 tokens/word cap
            text = clipped
        # yield word-ish chunks to exercise the SSE path
        for chunk in re.findall(r"\S+\s*", text):
            yield chunk


class XDna1Backend:
    """Real LLM engine on the XDNA1 host, with NPU GEMM offload.

    Base inference: Qwen2.5-0.5B (bf16, CPU/torch) — real model, real text.
    NPU: per-layer linear GEMMs are dispatched to the XDNA1 NPU via the IRON
    single_core bf16 matmul kernel during prefill (npu_gemm.py). The model id
    in models.json may carry a "model" field naming the HF repo.
    """

    name = "xdna1"

    def generate(self, model, messages, params):
        import engine

        eng = engine.get_engine(model.get("model"))
        yield from eng.generate(messages, params)


BACKENDS = {
    "mock": MockBackend(),
    "xdna1": XDna1Backend(),
}


# ---------------------------------------------------------------------------
# OpenAI response helpers
# ---------------------------------------------------------------------------


def count_tokens(text):
    """Rough token estimate (mock backends only; the real engine reports real
    counts). ~4 chars per token is fine for wiring tests."""
    return max(1, len(text) // 4) if text else 0


def _completion_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def chat_response(model_id, text, prompt_tokens, completion_tokens, finish_reason="stop"):
    return {
        "id": _completion_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chat_chunk(model_id, delta, finish_reason=None, usage=None):
    obj = {
        "id": _completion_id("chatcmpl"),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


def completion_response(model_id, text, prompt_tokens, completion_tokens, finish_reason="stop"):
    return {
        "id": _completion_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def completion_chunk(model_id, text, finish_reason=None):
    return {
        "id": _completion_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
    }


def error_response(message, err_type="invalid_request_error", code=None, status=400):
    err = {"message": message, "type": err_type}
    if code is not None:
        err["code"] = code
    return status, {"error": err}


# ---------------------------------------------------------------------------
# request handling
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "xdna-openai-shim/0.1"
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return None

    def _auth_ok(self):
        key = os.environ.get("XDNA_OAI_KEY")
        if not key:
            return True  # local dev: auth disabled
        auth = self.headers.get("Authorization", "")
        return auth == "Bearer " + key

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            if not self._auth_ok():
                status, obj = error_response("invalid API key", "authentication_error", 401)
                return self._send_json(obj, 401)
            data = [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "open-xdna",
                    "supports_chat": m.get("supports_chat", True),
                    **({"context_length": m["context_length"]}
                       if "context_length" in m else {}),
                }
                for m in self.server.models
            ]
            return self._send_json({"object": "list", "data": data})
        if path == "/health":
            resp = {
                "status": "ok",
                "backends": sorted(BACKENDS),
                "models": [m["id"] for m in self.server.models],
            }
            try:
                import engine  # noqa: F401
                if engine.get_engine.__code__ is not None:
                    pass
                if getattr(engine, "_engine", None) is not None:
                    resp["npu"] = engine._engine.npu_report()
                else:
                    import npu_gemm
                    resp["npu"] = {
                        "npu_available": npu_gemm.available(),
                        "npu_error": npu_gemm.error(),
                        "engine_loaded": False,
                    }
            except Exception as e:  # noqa: BLE001
                resp["npu"] = {"error": f"{type(e).__name__}: {e}"}
            # Rolling duty cycle: genuine NPU busy % for this platform
            # (kernel PMF sensor is unavailable on Hawk Point).
            try:
                resp["npu"]["npu_busy_percent"] = float(round(
                    self.server.npu_busy.percent(), 1
                ))
            except Exception:
                pass
            return self._send_json(resp)
        status, obj = error_response(f"unknown path {self.path}", "not_found", 404)
        return self._send_json(obj, 404)

    def do_POST(self):
        try:
            self._do_post()
        except Exception as e:  # noqa: BLE001 - surface as structured 500
            try:
                status, obj = error_response(
                    f"server error: {type(e).__name__}: {e}",
                    "server_error", 500, 500)
                self._send_json(obj, 500)
            except Exception:  # noqa: BLE001
                pass

    def _do_post(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._auth_ok():
            return self._send_json(error_response("invalid API key", "authentication_error")[1], 401)
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self._handle_chat()
        if path in ("/v1/completions", "/completions"):
            return self._handle_completions()
        status, obj = error_response(f"unknown path {self.path}", "not_found", 404)
        return self._send_json(obj, 404)

    # -- chat completions ----------------------------------------------------

    def _model_by_id(self, model_id):
        for m in self.server.models:
            if m["id"] == model_id:
                return m
        return None

    def _extract_params(self, req):
        return {
            "temperature": req.get("temperature"),
            "max_tokens": req.get("max_tokens") or req.get("max_completion_tokens"),
        }

    def _run_generation(self, model_id, messages, params):
        model = self._model_by_id(model_id)
        if model is None:
            raise RuntimeError(f"model '{model_id}' not found")
        backend = BACKENDS.get(model.get("backend", "mock"))
        if backend is None:
            raise RuntimeError(
                f"unknown backend '{model.get('backend')}' for model '{model_id}'")
        yield from backend.generate(model, messages, params)

    def _send_sse(self, frames, model_id):
        """frames: iterable of dict (or 'DONE'). Sends text/event-stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for frame in frames:
                if frame == "DONE":
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client (or lemond) went away mid-stream

    def _handle_chat(self):
        req = self._read_body()
        if req is None:
            status, obj = error_response("request body is not valid JSON")
            return self._send_json(obj, status)
        model_id = req.get("model", "")
        if self._model_by_id(model_id) is None:
            status, obj = error_response(
                f"model '{model_id}' not found", "model_not_found", 404)
            return self._send_json(obj, 404)
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            status, obj = error_response("'messages' must be a non-empty array")
            return self._send_json(obj, status)
        params = self._extract_params(req)
        stream = bool(req.get("stream", False))
        stream_options = req.get("stream_options") or {}
        include_usage = bool(stream_options.get("include_usage", False))

        prompt_tokens = sum(count_tokens(
            m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        ) for m in messages)

        if not stream:
            text = "".join(self._run_generation(model_id, messages, params))
            return self._send_json(chat_response(
                model_id, text, prompt_tokens, count_tokens(text)))

        def frames():
            full = []
            try:
                for delta in self._run_generation(model_id, messages, params):
                    full.append(delta)
                    yield chat_chunk(model_id, {"role": "assistant", "content": delta})
                yield chat_chunk(model_id, {}, finish_reason="stop")
            except NotImplementedError as e:
                yield {"error": {"message": str(e), "type": "backend_error"}}
                yield "DONE"
                return
            except Exception as e:  # noqa: BLE001 - surfaced as SSE error frame
                yield {"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}}
                yield "DONE"
                return
            text = "".join(full)
            if include_usage:
                # OpenAI's usage frame carries an EMPTY choices array; lemond's
                # is_usage_only_frame() swallow-check requires exactly that shape
                # when it injected include_usage itself.
                yield {
                    "id": _completion_id("chatcmpl"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": count_tokens(text),
                        "total_tokens": prompt_tokens + count_tokens(text),
                    },
                }
            yield "DONE"

        return self._send_sse(frames(), model_id)

    # -- legacy completions ----------------------------------------------------

    def _handle_completions(self):
        req = self._read_body()
        if req is None:
            status, obj = error_response("request body is not valid JSON")
            return self._send_json(obj, status)
        model_id = req.get("model", "")
        if self._model_by_id(model_id) is None:
            status, obj = error_response(
                f"model '{model_id}' not found", "model_not_found", 404)
            return self._send_json(obj, 404)
        prompt = req.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(str(p) for p in prompt)
        if not isinstance(prompt, str):
            status, obj = error_response("'prompt' must be a string or array of strings")
            return self._send_json(obj, status)
        params = self._extract_params(req)
        stream = bool(req.get("stream", False))
        messages = [{"role": "user", "content": prompt}]
        prompt_tokens = count_tokens(prompt)

        if not stream:
            text = "".join(self._run_generation(model_id, messages, params))
            return self._send_json(completion_response(
                model_id, text, prompt_tokens, count_tokens(text)))

        def frames():
            full = []
            try:
                for delta in self._run_generation(model_id, messages, params):
                    full.append(delta)
                    yield completion_chunk(model_id, delta)
                yield completion_chunk(model_id, "", finish_reason="stop")
            except Exception as e:  # noqa: BLE001
                yield {"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}}
                yield "DONE"
                return
            yield "DONE"

        return self._send_sse(frames(), model_id)


# ---------------------------------------------------------------------------


def main():
    host = os.environ.get("XDNA_OAI_HOST", "127.0.0.1")
    port = int(os.environ.get("XDNA_OAI_PORT", "8901"))
    models_path = os.environ.get(
        "XDNA_OAI_MODELS",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json"),
    )
    models = load_models(models_path)
    server = ThreadingHTTPServer((host, port), Handler)
    server.models = models
    server.npu_busy = NpuBusy()
    server.daemon_threads = True

    # Warm up the engine at startup so the NPU JIT-compiles every GEMM/attention
    # shape ONCE here, before the first real request. Without this, the first
    # chat request triggers a JIT storm (one compile per novel [M,K,N] shape)
    # and can take minutes. After warmup, cached kernels make requests fast.
    if os.environ.get("XDNA_NPU_WARMUP", "1") not in ("0", "false", "no"):
        try:
            import engine as _engine_mod
            print("xdna openai shim: warming up engine (NPU JIT compiles)...")
            _eng = _engine_mod.get_engine(models[0].get("model") if models else None)
            # one short dummy generation exercises prefill + decode shapes
            list(_eng.generate(
                [{"role": "user", "content": "warmup"}],
                {"max_tokens": 1, "temperature": 0.0, "top_k": 1},
            ))
            print(f"xdna openai shim: warmup done "
                  f"(jit_compiles={_engine_mod.npu_gemm.stats['compile_calls']}, "
                  f"npu_attention={'on' if _engine_mod._npu_attention_enabled() else 'off'})")
        except Exception as e:  # noqa: BLE001 - never block startup on warmup failure
            print(f"xdna openai shim: warmup skipped: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    print(f"xdna openai shim: listening on http://{host}:{port}")
    print(f"  models: {[m['id'] + ' (' + m['backend'] + ')' for m in models]}")
    print(f"  auth: {'required (XDNA_OAI_KEY set)' if os.environ.get('XDNA_OAI_KEY') else 'disabled'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
