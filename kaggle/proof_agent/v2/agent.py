"""ProofAgent v2 — offline Kaggle entry around the streaming pool loop.

Wires one StreamClient (tokenizer + http pool) + StreamingEngine (loop / time force-close)
+ ConcurrencyGate (prove/refine cap, verify priority), reused across problems. solve_pooled()
runs one problem under a wall-clock budget and always returns a best-available proof.
"""
from __future__ import annotations

import asyncio
import time
import zlib

from stream_engine import ConcurrencyGate, StreamClient, StreamingEngine
from pool_loop import solve_pooled as _solve_pooled

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prompts import fallback_preamble  # noqa: E402

_HARD_SLACK_S = 120.0


class ProofAgentV2:
    def __init__(self, base_url: str, model_path: str, *, temperature: float = 0.6,
                 top_p: float = 0.95, call_cap: int = 100_000, max_concurrent: int = 12,
                 gen_cap: int = 6, finalize_reserve_s: float = 180.0,
                 verify_temp: float = 0.6, select_temp: float = 0.6,
                 top_k: int | None = None):
        self.client = StreamClient(base_url, model_path, max_connections=max_concurrent + 8)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.call_cap = call_cap
        self.max_concurrent = max_concurrent
        self.gen_cap = gen_cap
        self.finalize_reserve_s = finalize_reserve_s
        self.role_temps = {"verify/": verify_temp, "select/": select_temp}

    async def health(self) -> bool:
        return await self.client.health()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def solve_pooled(self, problem: str, *, budget_s: float = 3300.0,
                           select_reserve_s: float = 600.0, init_provers: int = 6,
                           verify_k: int = 3, refine_inputs: int = 4, refine_min_seeds: int = 2,
                           select_bundle_n: int = 4, num_selectors: int = 5,
                           dump_path: str | None = None) -> dict:
        deadline = time.monotonic() + budget_s
        engine = StreamingEngine(self.client, temperature=self.temperature, top_p=self.top_p,
                                 call_cap=self.call_cap, max_tokens=self.call_cap,
                                 finalize_reserve_s=self.finalize_reserve_s,
                                 role_temps=self.role_temps,
                                 seed_base=zlib.crc32(problem.encode()) % 1_000_000,
                                 deadline=deadline, top_k=self.top_k)
        gate = ConcurrencyGate(total=self.max_concurrent, gen_cap=self.gen_cap)
        t0 = time.monotonic()
        coro = _solve_pooled(problem, engine, gate, deadline=deadline,
                             select_reserve_s=select_reserve_s, init_provers=init_provers,
                             verify_k=verify_k, refine_inputs=refine_inputs,
                             refine_min_seeds=refine_min_seeds,
                             select_bundle_n=select_bundle_n, num_selectors=num_selectors,
                             dump_path=dump_path)
        try:
            trace = await asyncio.wait_for(coro, timeout=budget_s + _HARD_SLACK_S)
        except asyncio.TimeoutError:
            trace = {"final_proof": fallback_preamble(), "final_source": "hard_timeout",
                     "selected_id": None, "selected_ids": [], "counts": {}, "totals": {}}
        trace["wall_s"] = time.monotonic() - t0
        return trace
