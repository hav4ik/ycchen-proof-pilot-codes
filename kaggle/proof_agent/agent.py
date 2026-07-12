"""ProofAgent: the offline Kaggle entry point around the DSMV2-Simple-3R loop.

Holds one LocalClient (tokenizer + http pool) reused across all problems. `solve()` runs the
full prove/verify/refine/select loop for ONE problem under a wall-clock budget and always returns
a best-available proof (the per-call deadline guard in Engine winds the pipeline down; a final
asyncio.wait_for is a last-resort net).

Usage:
    agent = ProofAgent("http://127.0.0.1:30000", "/path/to/model")
    trace = await agent.solve(problem, budget_s=3300)   # ~55 min, leaving margin under the 1h cap
    proof = trace["final_proof"]
    await agent.aclose()
"""
from __future__ import annotations

import asyncio
import time
import zlib

from client import LocalClient
from pipeline import Engine, solve_problem
from prompts import fallback_preamble

_HARD_SLACK_S = 120.0   # wait_for() net beyond the soft per-call deadline


class ProofAgent:
    def __init__(self, base_url: str, model_path: str, *, temperature: float = 0.6,
                 top_p: float = 0.95, max_tokens: int = 128_000, concurrency: int = 16,
                 est_tps: float = 35.0, call_cap: int = 32_000, salvage: bool = True,
                 salvage_tokens: int = 16_000, verify_temp: float = 1.0, select_temp: float = 0.2,
                 top_k: int | None = None):
        self.client = LocalClient(base_url, model_path, temperature=temperature, top_p=top_p,
                                  max_connections=concurrency + 8, top_k=top_k)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.est_tps = est_tps
        self.call_cap = call_cap
        self.salvage = salvage
        self.salvage_tokens = salvage_tokens
        # B4: low temp for the discrete roles (stable scores / format-compliant id picks)
        self.role_temps = {"verify/": verify_temp, "select/": select_temp}

    async def health(self) -> bool:
        return await self.client.health()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def solve(self, problem: str, *, budget_s: float | None = 3300.0, num_provers: int = 6,
                    verify_k: int = 2, num_refiners: int = 3, num_selectors: int = 4) -> dict:
        sem = asyncio.Semaphore(self.concurrency)
        deadline = None if budget_s is None else time.monotonic() + budget_s
        engine = Engine(self.client, sem, max_tokens=self.max_tokens, temperature=self.temperature,
                        top_p=self.top_p, deadline=deadline, est_tps=self.est_tps,
                        call_cap=self.call_cap, salvage=self.salvage, salvage_tokens=self.salvage_tokens,
                        role_temps=self.role_temps, seed_base=zlib.crc32(problem.encode()) % 1_000_000)
        t0 = time.monotonic()
        coro = solve_problem(problem, engine, num_provers=num_provers, verify_k=verify_k,
                             num_refiners=num_refiners, num_selectors=num_selectors)
        try:
            hard = None if budget_s is None else budget_s + _HARD_SLACK_S
            trace = await asyncio.wait_for(coro, timeout=hard)
        except asyncio.TimeoutError:
            trace = {"stages": {}, "final_proof": fallback_preamble(),
                     "final_source": "hard_timeout", "selected_id": None, "selected_ids": [],
                     "counts": {}, "totals": {}}
        trace["wall_s"] = time.monotonic() - t0
        return trace

    async def solve_pooled(self, problem: str, *, budget_s: float = 3300.0,
                           select_reserve_s: float = 600.0, concurrency: int | None = None,
                           init_provers: int = 6, verify_k: int = 3, refine_inputs: int = 4,
                           select_bundle_n: int = 4, num_selectors: int = 5,
                           dump_path: str | None = None) -> dict:
        """Continuous pool loop (pipelined; fills the budget). The pool owns concurrency via its own
        semaphore, so the Engine's semaphore is effectively unbounded here."""
        from pool_loop import solve_pooled as _solve_pooled
        deadline = time.monotonic() + budget_s
        engine = Engine(self.client, asyncio.Semaphore(10_000), max_tokens=self.max_tokens,
                        temperature=self.temperature, top_p=self.top_p, deadline=deadline,
                        est_tps=self.est_tps, call_cap=self.call_cap, salvage=self.salvage,
                        salvage_tokens=self.salvage_tokens, role_temps=self.role_temps,
                        seed_base=zlib.crc32(problem.encode()) % 1_000_000)
        t0 = time.monotonic()
        coro = _solve_pooled(problem, engine, deadline=deadline,
                             select_reserve_s=select_reserve_s,
                             concurrency=concurrency or self.concurrency,
                             init_provers=init_provers, verify_k=verify_k,
                             refine_inputs=refine_inputs, select_bundle_n=select_bundle_n,
                             num_selectors=num_selectors, dump_path=dump_path)
        # A1 (last-resort net): the pool's internal _select_phase wait_for should keep us inside the
        # budget, but wrap the whole thing too so Kaggle's 1h cap can never be breached.
        try:
            trace = await asyncio.wait_for(coro, timeout=budget_s + _HARD_SLACK_S)
        except asyncio.TimeoutError:
            trace = {"final_proof": fallback_preamble(), "final_source": "hard_timeout",
                     "selected_id": None, "selected_ids": [], "counts": {}, "totals": {}}
        trace["wall_s"] = time.monotonic() - t0
        return trace
