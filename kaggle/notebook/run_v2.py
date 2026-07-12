"""Kaggle submission entry point for Proof Pilot — **v2 streaming** pool loop.

Reads the competition CSV, runs the offline streaming prove/verify/refine/select loop
(proof_agent/v2 — real-time loop-detect + time force-close + salvage) against a LOCAL sglang
server, ONE problem at a time (sequential; the single server is reused across problems), and
writes `id,answer` (matching the competition `sample_submission.csv` header) where `answer` is
the final proof text.

Differences vs the v1 run.py:
  - engine is ProofAgentV2 (streaming SSE; no est_tps — the force-close continuation is sized
    from the live char-rate). call_cap defaults to 100k (v2), not 32k.
  - output column is `answer`, not `prediction`.
  - ALL problem ids are pre-seeded with a fallback answer up front and overwritten as each is
    solved, so a crash / Kaggle wall-clock kill still yields a complete (every-id-present)
    submission.csv rather than a short file the grader would reject.

Per-problem time governance (the central Kaggle risk):
  - budget_s   : wall-clock per problem (default 4200 = 70 min; Kaggle 1h + 30 min grace).
  - deadline   : optional overall epoch-sec deadline (0 = off); per-problem budget is shrunk to
                 min(budget_s, remaining/problems_left) so a global limit is never blown either.
  v2's streaming watchdog + force-close-think + fallback chain guarantee a best-available answer.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# v2 package dir FIRST: `agent`/`pool_loop`/`stream_engine`/`loopguard` resolve to v2; the v2
# modules self-append the parent proof_agent dir for the shared parser/bundle/clean/prompts/rank.
sys.path.insert(0, str(REPO / "kaggle" / "proof_agent" / "v2"))
from agent import ProofAgentV2  # noqa: E402

MAX_PROOF_CHARS = 18000   # ~5 pages of text safety clip (no figures; pure text)
FALLBACK_ANSWER = ("We were unable to produce a complete proof within the time limit.")


def read_problems(path: Path) -> list[dict]:
    """Accept a CSV with an id column (id/problem_id/ID) and a problem column
    (problem/question/statement)."""
    rows = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            pid = r.get("id") or r.get("problem_id") or r.get("ID") or str(i)
            prob = r.get("problem") or r.get("question") or r.get("statement") or ""
            rows.append({"id": pid, "problem": prob})
    return rows


def clip_proof(text: str) -> str:
    text = (text or "").strip()
    if len(text) > MAX_PROOF_CHARS:
        text = text[:MAX_PROOF_CHARS].rsplit("\n", 1)[0] + "\n\n[truncated to length limit]"
    return text


def write_output(path: str, answers: dict[str, str], order: list[str]) -> None:
    """Write `id,answer` for EVERY id (in input order). Called after every problem so a partial
    run still leaves a complete, grader-valid submission."""
    p = Path(path)
    if p.suffix != ".csv":           # some harnesses pass an output DIR
        p.mkdir(parents=True, exist_ok=True); p = p / "submission.csv"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "answer"])
        for pid in order:
            w.writerow([pid, answers[pid]])


async def run(args) -> None:
    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
    trace_f = open(logdir / "traces.jsonl", "w")
    problems = read_problems(Path(args.input_csv))
    order = [row["id"] for row in problems]
    # pre-seed every id with the fallback so submission.csv is always complete (see module docstring)
    answers: dict[str, str] = {pid: FALLBACK_ANSWER for pid in order}
    write_output(args.output_csv, answers, order)
    print(f"[run_v2] {len(problems)} problems | base={args.base_url} budget_s={args.budget_s} "
          f"-> {args.output_csv} (pre-seeded fallback)", flush=True)

    agent = ProofAgentV2(args.base_url, args.model_path, temperature=args.temperature,
                         top_p=args.top_p, call_cap=args.call_cap,
                         max_concurrent=args.concurrency, gen_cap=args.gen_cap,
                         finalize_reserve_s=args.finalize_reserve,
                         verify_temp=args.verify_temp, select_temp=args.select_temp,
                         top_k=args.top_k)
    # wait for the server (the notebook launched it just before us)
    for _ in range(args.health_wait // 5):
        if await agent.health():
            break
        await asyncio.sleep(5)

    for i, row in enumerate(problems):
        left = len(problems) - i
        budget = args.budget_s
        if args.deadline:
            remaining = args.deadline - time.time()
            budget = max(60.0, min(budget, remaining / max(1, left)))
        t0 = time.time()
        try:
            tr = await agent.solve_pooled(
                row["problem"], budget_s=budget, select_reserve_s=args.select_reserve,
                init_provers=args.init_provers, verify_k=args.verify_k,
                refine_inputs=args.refine_inputs, refine_min_seeds=args.refine_min_seeds,
                select_bundle_n=args.select_bundle_n, num_selectors=args.selectors,
                dump_path=str(logdir / f"trace_{row['id']}.json"))
            proof = clip_proof(tr.get("final_proof"))
            src = tr.get("final_source")
        except Exception:  # noqa: BLE001 — never let one problem sink the submission
            proof = FALLBACK_ANSWER
            src = "exception"
            traceback.print_exc()
            tr = {"counts": {}, "totals": {}}
        if proof.strip():
            answers[row["id"]] = proof
        trace_f.write(json.dumps({"id": row["id"], "final_source": src, "wall_s": time.time() - t0,
                                  "counts": tr.get("counts"), "totals": tr.get("totals"),
                                  "proof_len": len(proof)}, ensure_ascii=False) + "\n")
        trace_f.flush()
        print(f"[run_v2] {i+1}/{len(problems)} id={row['id']} wall={time.time()-t0:.0f}s "
              f"src={src} len={len(proof)}", flush=True)
        # rewrite the full submission incrementally (every id always present)
        write_output(args.output_csv, answers, order)

    trace_f.close()
    await agent.aclose()
    print(f"[run_v2] wrote {len(order)} answers -> {args.output_csv}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="target model dir (tokenizer + served name)")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--logdir", default="/tmp/proof_logs")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--budget-s", type=float, default=4200.0, help="wall-clock per problem (70min; Kaggle 1h + 30min grace)")
    ap.add_argument("--deadline", type=float, default=0.0, help="overall epoch-sec deadline (0=off)")
    # pool-loop params (v2 — see proof_agent/v2/DESIGN.md)
    ap.add_argument("--select-reserve", type=float, default=900.0, help="reserve window for the selector vote (15min)")
    ap.add_argument("--init-provers", type=int, default=6)
    ap.add_argument("--verify-k", type=int, default=3)
    ap.add_argument("--refine-inputs", type=int, default=4, help="candidates merged per refine")
    ap.add_argument("--refine-min-seeds", type=int, default=2,
                    help="min verified candidates to merge-refine (else spawn a prove)")
    ap.add_argument("--select-bundle-n", type=int, default=4)
    ap.add_argument("--selectors", type=int, default=5)
    # engine knobs (v2)
    ap.add_argument("--call-cap", type=int, default=100000, help="per-call token ceiling (v2 default)")
    ap.add_argument("--concurrency", type=int, default=12, help="total concurrent calls (gate total)")
    ap.add_argument("--gen-cap", type=int, default=6, help="prove/refine sub-cap (verify gets priority)")
    ap.add_argument("--finalize-reserve", type=float, default=180.0, help="still in <think> with < this left -> force-close")
    ap.add_argument("--temperature", type=float, default=0.6, help="prover/refiner temp")
    ap.add_argument("--verify-temp", type=float, default=0.6)
    ap.add_argument("--select-temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--health-wait", type=int, default=1200, help="seconds to wait for server")
    # FMI-required knobs (submissions-instructions §2a). Aliases onto the native knobs above so
    # behavior is unchanged when unset.
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="FMI alias for --call-cap: per-call generation ceiling")
    ap.add_argument("--num_ctx", type=int, default=None,
                    help="FMI knob: context bound; clamps the per-call ceiling here. The server "
                         "context length itself is set at serve time (serve/serve_final.sh CTX)")
    ap.add_argument("--top_k", type=int, default=None,
                    help="FMI knob: top-k sampling (unset = server default, i.e. disabled)")
    args = ap.parse_args()
    if args.max_new_tokens is not None:
        args.call_cap = args.max_new_tokens
    if args.num_ctx is not None:
        args.call_cap = min(args.call_cap, args.num_ctx)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
