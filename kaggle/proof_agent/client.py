"""Local sglang client for the offline Kaggle proof agent.

Replaces distill_gen's DeepSeek AsyncChatClient. Talks to a local sglang server
(OpenAI-compatible /v1/chat/completions for normal calls; native /generate for the
force-close-think salvage which needs token-space input). All async via httpx so the
pipeline's asyncio.gather wave-concurrency is preserved.

Key differences vs the DeepSeek client (see distill_gen notes):
  - NO reasoning_effort (DeepSeek-only; it also strips temperature). We send explicit
    temperature/top_p — OLMo reasoning models loop at T=0, so diversity needs sampling.
  - reasoning_content vs content: sglang with `--reasoning-parser deepseek-r1` splits the
    <think> CoT into message.reasoning_content and the post-think answer into message.content.
    The pipeline parses ONLY content for <solution>/<score>/<selected_id>.
"""
from __future__ import annotations

import time

import httpx
from transformers import AutoTokenizer


class LocalClient:
    def __init__(self, base_url: str, model_path: str, *, temperature: float = 0.6,
                 top_p: float = 0.95, request_timeout: float = 3600.0,
                 max_connections: int = 64, top_k: int | None = None):
        self.base = base_url.rstrip("/")
        self.model = model_path  # sglang served_model_name == the model path
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        limits = httpx.Limits(max_connections=max_connections,
                              max_keepalive_connections=max_connections)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(request_timeout), limits=limits)
        # tokenizer for the salvage path (apply_chat_template + encode). Offline-safe.
        self.tok = AutoTokenizer.from_pretrained(model_path)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(self, messages: list[dict], *, max_tokens: int,
                   temperature: float | None = None, top_p: float | None = None,
                   timeout: float | None = None, seed: int | None = None) -> dict:
        """One OpenAI-compatible chat call. Returns the Engine's expected record shape."""
        t0 = time.monotonic()
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k   # sglang chat-completions extension
        if seed is not None:
            payload["seed"] = seed   # sglang honours OpenAI `seed` -> reproducible per-call sampling
        r = await self._http.post(f"{self.base}/v1/chat/completions", json=payload,
                                  timeout=timeout)
        r.raise_for_status()
        j = r.json()
        choice = j["choices"][0]
        msg = choice.get("message", {}) or {}
        usage = j.get("usage", {}) or {}
        details = usage.get("completion_tokens_details") or {}
        return {
            "message": {"content": msg.get("content") or "",
                        "reasoning_content": msg.get("reasoning_content") or ""},
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "latency_s": time.monotonic() - t0,
        }

    async def generate_raw(self, input_ids: list[int], *, max_new_tokens: int,
                           temperature: float | None = None, top_p: float | None = None,
                           timeout: float | None = None, seed: int | None = None) -> dict:
        """Native sglang /generate over explicit input_ids (no chat template, no double-BOS).
        Used by the salvage path to continue from a truncated chain-of-thought."""
        # NOTE: sglang's native /generate sampling_params rejects `seed` (HTTP 500) on this build,
        # unlike /v1/chat/completions which accepts a top-level seed. `seed` is accepted here for a
        # uniform signature but deliberately NOT forwarded — sending it 500s and kills salvage.
        sp = {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "max_new_tokens": max_new_tokens,
        }
        if self.top_k is not None:
            sp["top_k"] = self.top_k
        payload = {"input_ids": input_ids, "sampling_params": sp}
        r = await self._http.post(f"{self.base}/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, list):
            out = out[0]
        return out  # {"text": ..., "meta_info": {"finish_reason": ..., "completion_tokens": ...}}

    async def health(self) -> bool:
        try:
            r = await self._http.get(f"{self.base}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False
