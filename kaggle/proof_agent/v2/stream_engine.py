"""Streaming generation engine (v2) + concurrency gate.

See DESIGN.md. One `StreamingEngine.generate()` opens an SSE stream to the local sglang
server and applies a stop policy as tokens arrive, so it can stop EARLY on a loop or when
the call deadline approaches while still inside <think> (the "given the time ... </think>"
force-close). The returned record matches the v1 Engine shape so parser/bundle/clean are
reused unchanged.

No est_tps: the force-close continuation is sized from THIS stream's measured char-rate.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time

import httpx

from loopguard import find_loop_cut, recent_window
from zlib_runaway_detector import RunawayDetector, scan, WINDOW_CHARS, STEP_CHARS

# ---- stop-policy constants (wall-clock knobs only; NO token/time extrapolation constant) ----
_MIN_CALL_S = 30.0          # below this much call-time left, don't even start
_FINALIZE_RESERVE_S = 180.0  # still in <think> with less than this left -> force-close now
_CHECK_EVERY_CHARS = 2000   # run the loop/time checks each time this many new chars arrive
_LOOP_WINDOW = 16_000       # scan only the last N chars for a loop
_MIN_SALVAGE_TOK = 2048
_MAX_SALVAGE_TOK = 4096   # salvage is a short finalize, not a re-derivation; a long salvage just rambles/loops
_FALLBACK_TOK_PER_S = 30.0  # only used if a stream aborts before we have a rate sample


class StreamClient:
    """httpx transport: SSE streaming chat, native /generate (force-close), abort, tokenizer."""

    def __init__(self, base_url: str, model_path: str, *, max_connections: int = 64):
        self.base = base_url.rstrip("/")
        self.model = model_path
        limits = httpx.Limits(max_connections=max_connections,
                              max_keepalive_connections=max_connections)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=limits)
        from transformers import AutoTokenizer   # lazy: only needed for the salvage token path
        self.tok = AutoTokenizer.from_pretrained(model_path)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def health(self) -> bool:
        try:
            r = await self._http.get(f"{self.base}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def stream_chat(self, messages, *, max_tokens, temperature, top_p, seed, timeout,
                          top_k=None):
        """Async-generate SSE events: dicts with optional rid / reasoning / content / finish /
        usage. The caller drives this and decides when to stop (breaking the loop closes the
        stream -> sglang aborts the running request)."""
        payload = {
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "stream": True,
            "stream_options": {"include_usage": True},
        }
        if top_k is not None:
            payload["top_k"] = top_k   # sglang chat-completions extension
        if seed is not None:
            payload["seed"] = seed
        async with self._http.stream("POST", f"{self.base}/v1/chat/completions",
                                     json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                ev = {"rid": j.get("id"), "usage": j.get("usage")}
                choices = j.get("choices") or []
                if choices:
                    ch = choices[0]
                    delta = ch.get("delta") or {}
                    ev["reasoning"] = delta.get("reasoning_content")
                    ev["content"] = delta.get("content")
                    ev["finish"] = ch.get("finish_reason")
                yield ev

    async def generate_raw(self, input_ids, *, max_new_tokens, temperature, top_p, timeout,
                           top_k=None):
        """Native /generate over explicit input_ids (force-close continuation). NOTE: this
        build's /generate 500s on `seed`, so seed is never forwarded here (see v1 client.py)."""
        sp = {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens}
        if top_k is not None:
            sp["top_k"] = top_k
        payload = {"input_ids": input_ids, "sampling_params": sp}
        r = await self._http.post(f"{self.base}/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        out = r.json()
        return out[0] if isinstance(out, list) else out

    async def abort(self, rid: str | None) -> None:
        if not rid:
            return
        with contextlib.suppress(Exception):
            await self._http.post(f"{self.base}/abort_request", json={"rid": rid}, timeout=5.0)


class ConcurrencyGate:
    """Total concurrency = `total`; prove/refine sub-capped at `gen_cap`; verify has priority.

    gen() (prove/refine) admits only when under gen_cap, under total, AND no verify is
    waiting (-> verify pre-empts prove/refine for the next freed slot). verify() admits
    whenever under total. Since gen_cap < total, >=(total-gen_cap) slots are structurally
    reserved for verify even before the priority rule.
    """

    def __init__(self, total: int = 12, gen_cap: int = 8):
        self.total = total
        self.gen_cap = gen_cap
        self._tot = 0
        self._gen = 0
        self._vw = 0                       # verifiers waiting
        self._cond = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def gen(self):
        async with self._cond:
            await self._cond.wait_for(
                lambda: self._gen < self.gen_cap and self._tot < self.total and self._vw == 0)
            self._gen += 1
            self._tot += 1
        try:
            yield
        finally:
            async with self._cond:
                self._gen -= 1
                self._tot -= 1
                self._cond.notify_all()

    @contextlib.asynccontextmanager
    async def verify(self):
        async with self._cond:
            self._vw += 1
            self._cond.notify_all()        # let any checking gen see a verify is waiting -> yield
            try:
                await self._cond.wait_for(lambda: self._tot < self.total)
            finally:
                self._vw -= 1
            self._tot += 1
        try:
            yield
        finally:
            async with self._cond:
                self._tot -= 1
                self._cond.notify_all()


class StreamingEngine:
    """One generate() == one streamed sglang call with the loop / time-forceclose stop policy."""

    _ROLE_TAG = {"prove/": "<solution>", "refine/": "<solution>",
                 "verify/": "<score>", "select/": "<selected_id>"}

    def __init__(self, client: StreamClient, *, temperature: float = 1.0, top_p: float = 0.95,
                 call_cap: int = 100_000, max_tokens: int = 100_000,
                 finalize_reserve_s: float = _FINALIZE_RESERVE_S,
                 role_temps: dict | None = None, seed_base: int = 1234,
                 deadline: float | None = None, top_k: int | None = None):
        self.client = client
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.call_cap = call_cap
        self.max_tokens = max_tokens
        self.finalize_reserve_s = finalize_reserve_s
        self.role_temps = role_temps or {}
        self.seed_base = seed_base
        self._seed_ctr = 0
        self.deadline = deadline           # monotonic; per-call window end

    # ---- helpers ----
    def _temp_for(self, label: str) -> float:
        for prefix, t in self.role_temps.items():
            if label.startswith(prefix):
                return t
        return self.temperature

    def _remaining(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.monotonic()

    def _blank(self, label) -> dict:
        return {"label": label, "reasoning_content": "", "content": "", "finish_reason": None,
                "truncated": False, "error": None, "salvaged": False, "stop_reason": None,
                "seed": None, "temperature": None, "max_tokens": None, "rid": None,
                "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                "latency_s": None, "t_start": None, "t_end": None}

    def _needs_tag(self, label: str, content: str) -> bool:
        tag = next((t for p, t in self._ROLE_TAG.items() if label.startswith(p)), None)
        return bool(tag) and tag not in (content or "").lower()

    # ---- main ----
    async def generate(self, messages, *, label: str, t0: float | None = None) -> dict:
        rec = self._blank(label)
        t_start = time.monotonic()
        rec["t_start"] = (t_start - t0) if t0 is not None else None
        remaining = self._remaining()
        if remaining <= _MIN_CALL_S:
            rec.update(finish_reason="timeout", error="deadline: no time left",
                       t_end=(time.monotonic() - t0) if t0 is not None else None)
            return rec

        temp = self._temp_for(label)
        seed = self.seed_base + self._seed_ctr
        self._seed_ctr += 1
        cap = min(self.max_tokens, self.call_cap)
        rec.update(seed=seed, temperature=temp, max_tokens=cap)
        is_solution_role = label.startswith("prove/") or label.startswith("refine/")

        reasoning, content = [], []
        rlen = clen = 0
        rid = None
        usage = None
        finish = None
        stop_reason = None
        loop_verdict = None
        det = RunawayDetector()             # zlib sliding-window loop detector, fed the live text
        chars_since_check = 0

        try:
            stream = self.client.stream_chat(messages, max_tokens=cap, temperature=temp,
                                             top_p=self.top_p, seed=seed, top_k=self.top_k,
                                             timeout=remaining + 30.0)
            async for ev in stream:
                rid = ev.get("rid") or rid
                if ev.get("usage"):
                    usage = ev["usage"]
                # (1) LOOP — feed the live text (reasoning, then content) to the zlib detector.
                # It self-paces (checks every STEP_CHARS over a WINDOW_CHARS window); HARD trips
                # on degenerate token loops at once, SOFT on sustained semantic near-loops.
                aborted = False
                for txt, sink, is_reason in ((ev.get("reasoning"), reasoning, True),
                                            (ev.get("content"), content, False)):
                    if not txt:
                        continue
                    sink.append(txt)
                    if is_reason:
                        rlen += len(txt)
                    else:
                        clen += len(txt)
                    chars_since_check += len(txt)
                    v = det.feed(txt)
                    if v.abort:
                        stop_reason = "loop"; loop_verdict = v; aborted = True
                        break
                if aborted:
                    break
                if ev.get("finish"):
                    finish = ev["finish"]
                    break
                # (2) TIME force-close — only while still thinking (no <solution> in content yet)
                if chars_since_check >= _CHECK_EVERY_CHARS:
                    chars_since_check = 0
                    cstr = "".join(content)
                    if "<solution>" not in cstr.lower() and self._remaining() < self.finalize_reserve_s:
                        stop_reason = "time_forceclose"
                        break
            # close the stream (client disconnect aborts the request) + explicit abort
            with contextlib.suppress(Exception):
                await stream.aclose()
            if stop_reason:
                await self.client.abort(rid)

            rstr, cstr = "".join(reasoning), "".join(content)
            rec.update(rid=rid, reasoning_content=rstr, content=cstr, finish_reason=finish)
            if usage:
                rt = usage.get("reasoning_tokens")
                if rt is None:
                    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                rec.update(prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"), reasoning_tokens=rt)
            else:
                rec["completion_tokens"] = (rlen + clen) // 4   # estimate when aborted (no usage chunk)

            # ---- salvage paths ----
            if stop_reason == "loop":
                await self._salvage_loop(rec, messages, label, rstr, cstr, t_start, rlen + clen,
                                         loop_verdict)
            elif stop_reason == "time_forceclose":
                await self._salvage_time(rec, messages, label, rstr, t_start, rlen + clen)
            elif finish == "length" and "<solution>" not in cstr.lower():
                # hit the per-call token cap while still in <think> (no <solution> written yet):
                # force-close a solution from the clean reasoning — same mechanism as the time path.
                # Without this the truncated CoT yields no <solution> and the whole call is discarded
                # (plen=0), which makes a small call_cap useless (every prove dies mid-reasoning).
                await self._salvage_time(rec, messages, label, rstr, t_start, rlen + clen,
                                         reason="length_forceclose")
            else:
                rec["truncated"] = finish == "length"
                rec["stop_reason"] = finish

        except Exception as e:  # noqa: BLE001 — never raise; record the cause chain
            parts, cur, seen = [], e, set()
            while cur is not None and id(cur) not in seen:
                seen.add(id(cur)); parts.append(repr(cur)); cur = cur.__cause__ or cur.__context__
            rec.update(finish_reason="error", error=" <- ".join(parts))

        rec["latency_s"] = time.monotonic() - t_start
        rec["t_end"] = (time.monotonic() - t0) if t0 is not None else None
        return rec

    def _live_tok_per_s(self, total_chars: int, t_start: float) -> float:
        elapsed = max(1e-3, time.monotonic() - t_start)
        rate = (total_chars / 4.0) / elapsed
        return rate if rate > 1.0 else _FALLBACK_TOK_PER_S

    def _salvage_cap(self, total_chars: int, t_start: float) -> int:
        # size the continuation to what actually fits in the remaining time at THIS stream's live
        # rate — NEVER floor above the time budget (a too-large cap overruns the /generate timeout
        # and the whole salvage is lost). Small floor so a tight reserve still writes a short proof.
        rem = self._remaining()
        if rem <= _MIN_CALL_S:
            return 512
        by_time = int((rem - 10.0) * self._live_tok_per_s(total_chars, t_start))
        return max(256, min(_MAX_SALVAGE_TOK, by_time))

    async def _force_close(self, messages, reasoning_prefix: str, steer: str, *,
                           cap: int, temp: float) -> dict:
        sys_user = [m for m in messages if m.get("role") in ("system", "user")]
        prefix = self.client.tok.apply_chat_template(sys_user, add_generation_prompt=True,
                                                     tokenize=True, return_dict=False)
        if hasattr(prefix, "keys"):
            prefix = prefix["input_ids"]
        if prefix and isinstance(prefix[0], list):
            prefix = prefix[0]
        cont = self.client.tok.encode((reasoning_prefix or "") + steer, add_special_tokens=False)
        rem = self._remaining()
        return await self.client.generate_raw(list(prefix) + cont, max_new_tokens=cap,
                                               temperature=temp, top_p=self.top_p,
                                               top_k=self.top_k,
                                               timeout=(rem - 5.0 if rem != float("inf") else 600.0))

    def _apply_salvage(self, rec, out, label):
        text = (out.get("text") or "")
        if not text.strip():
            return
        content = text if "<solution>" in text.lower() else "<solution>\n" + text
        # guard: a force-close from a still-looping context can itself loop. If so, cut to the clean
        # prefix — that prefix is short, so the parser's MIN_SOLUTION_CHARS gate drops it (a loop must
        # NOT enter the pool as a "valid" proof).
        if scan(content).abort:
            cut = find_loop_cut(content)
            if cut is not None:
                content = content[:cut]
        rec["content"] = content
        cont_fr = (out.get("meta_info") or {}).get("finish_reason")
        if isinstance(cont_fr, dict):
            cont_fr = cont_fr.get("type")
        rec["truncated"] = cont_fr == "length"
        # A salvaged proof is judged on its recovered <solution>, NOT on the continuation hitting
        # its own cap — else the parser's finish!=length rule vetoes a usable (if truncated)
        # recovered proof and wastes it. Mark stop when there's a real solution body.
        usable = "<solution>" in content.lower() and len(content) > 500
        rec["finish_reason"] = "stop" if usable else (cont_fr or "stop")
        rec["salvaged"] = True

    def _loop_onset(self, text: str, verdict) -> int:
        """Where to truncate a looping text to keep the clean pre-loop prefix.
        Char-precise cut for verbatim loops; for a semantic loop (no verbatim cut) fall back to
        the zlib detector's onset estimate — position minus the repetitive window and the
        sustained-soft run that preceded the abort."""
        cut = find_loop_cut(text)
        if cut is not None:
            return cut
        if verdict is not None and verdict.position:
            onset = verdict.position - WINDOW_CHARS - verdict.soft_run * STEP_CHARS
            return max(0, min(len(text), onset))
        return len(text)

    async def _salvage_loop(self, rec, messages, label, reasoning, content, t_start, total_chars,
                            verdict=None):
        """Loop aborted. If the loop is in the SOLUTION (content), just truncate it — the proof
        is the clean prefix. If it's in the reasoning, drop the looping tail and force-close."""
        rec["stop_reason"] = "loop"
        if content.strip():
            cut = find_loop_cut(recent_window(content, _LOOP_WINDOW))
            # map window-relative cut back, conservatively keep everything before the window tail
            if cut is not None:
                base = max(0, len(content) - _LOOP_WINDOW)
                rec["content"] = content[:base + cut]
                rec["salvaged"] = True
                rec["finish_reason"] = "stop"
                return
        # loop in reasoning -> keep clean prefix (verbatim cut, else zlib onset), force-close fresh
        clean = reasoning[: self._loop_onset(reasoning, verdict)]
        steer = ("\n\nI will finalize now and write ONLY the rigorous proof itself below — no "
                 "planning, no meta-commentary, just the mathematics.\n</think>\n\n<solution>\n")
        cap = self._salvage_cap(total_chars, t_start)
        try:
            out = await self._force_close(messages, clean, steer, cap=cap, temp=self._temp_for(label))
            self._apply_salvage(rec, out, label)
        except Exception as e:  # noqa: BLE001 — salvage best-effort, but surface why it failed
            rec["salvage_error"] = repr(e)

    async def _salvage_time(self, rec, messages, label, reasoning, t_start, total_chars,
                            reason: str = "time_forceclose"):
        """Force-close while still in <think>: append a 'finalize now' steer and let the model write
        the solution from its (incomplete) reasoning. Triggered by the call deadline approaching
        (reason='time_forceclose') OR the per-call token cap being hit mid-think ('length_forceclose')."""
        rec["stop_reason"] = reason
        # Strong steer: a model force-closed BEFORE it converged tends to keep "planning" in the
        # solution space ("We need to produce a final answer ...") instead of writing the proof.
        # Tell it explicitly to emit only the proof body.
        steer = ("\n\nI am out of time and must finalize now. I will write ONLY the rigorous proof "
                 "itself below — no planning, no meta-commentary, no restating the task, just the "
                 "mathematics.\n</think>\n\n<solution>\n")
        cap = self._salvage_cap(total_chars, t_start)
        try:
            out = await self._force_close(messages, reasoning, steer, cap=cap, temp=self._temp_for(label))
            self._apply_salvage(rec, out, label)
        except Exception as e:  # noqa: BLE001 — salvage best-effort, but surface why it failed
            rec["salvage_error"] = repr(e)
