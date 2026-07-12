"""Kaggle / FMI run-variant entry point for the Proof Pilot agentic proof system.

Reads a problem CSV, runs the offline prove/verify/refine/select loop (proof_agent) against a
LOCAL sglang server (one problem at a time — the Kaggle setting), and writes `id,prediction` where
prediction is the final proof text. FMI-compatible flags: --model_path --input_csv --output_csv
--logdir. The notebook launches serve_final.sh first and passes --base-url here.

Per-problem time governance (the central Kaggle risk):
  - budget_s   : wall-clock per problem (default 3300 = 55 min, leaving margin under the 1h cap).
  - deadline   : optional overall notebook deadline (epoch sec); per-problem budget is shrunk to
                 min(budget_s, remaining/problems_left) so we never blow a global limit either.
  The proof_agent watchdog + force-close-think + fallback chain guarantee a best-available proof.
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
sys.path.insert(0, str(REPO / "kaggle" / "proof_agent"))
from agent import ProofAgent  # noqa: E402

MAX_PROOF_CHARS = 18000   # ~5 pages of text safety clip (no figures; pure text)


def read_problems(path: Path) -> list[dict]:
    """Accept CSV with columns including an id (id/problem_id/ID) and a problem (problem/question)."""
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


async def run(args) -> None:
    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
    trace_f = open(logdir / "traces.jsonl", "w")
    problems = read_problems(Path(args.input_csv))
    print(f"[run] {len(problems)} problems | base={args.base_url} budget_s={args.budget_s}", flush=True)

    agent = ProofAgent(args.base_url, args.model_path, temperature=args.temperature,
                       max_tokens=args.max_tokens, concurrency=args.concurrency,
                       est_tps=args.est_tps, call_cap=args.call_cap, top_k=args.top_k,
                       top_p=args.top_p)
    # wait for the server (it was launched by the notebook just before us)
    for _ in range(args.health_wait // 5):
        if await agent.health():
            break
        await asyncio.sleep(5)

    results: list[tuple[str, str]] = []
    for i, row in enumerate(problems):
        left = len(problems) - i
        budget = args.budget_s
        if args.deadline:
            remaining = args.deadline - time.time()
            budget = max(60.0, min(budget, remaining / max(1, left)))
        t0 = time.time()
        try:
            # validated path: continuous pool loop (fills the budget; pipelined prove/verify/refine,
            # selector vote in the reserved window). See rehearsal/RESULTS.md.
            tr = await agent.solve_pooled(row["problem"], budget_s=budget,
                                          select_reserve_s=args.select_reserve,
                                          init_provers=args.init_provers, verify_k=args.verify_k,
                                          refine_inputs=args.refine_inputs,
                                          select_bundle_n=args.select_bundle_n,
                                          num_selectors=args.selectors,
                                          dump_path=str(logdir / f"trace_{row['id']}.json"))
            proof = clip_proof(tr["final_proof"])
            src = tr["final_source"]
        except Exception:  # noqa: BLE001 — never let one problem sink the submission
            proof = "We were unable to produce a proof within the time limit."
            src = "exception"
            traceback.print_exc()
            tr = {"counts": {}, "totals": {}}
        results.append((row["id"], proof))
        trace_f.write(json.dumps({"id": row["id"], "final_source": src, "wall_s": time.time() - t0,
                                  "counts": tr.get("counts"), "totals": tr.get("totals"),
                                  "proof_len": len(proof)}, ensure_ascii=False) + "\n")
        trace_f.flush()
        print(f"[run] {i+1}/{len(problems)} id={row['id']} wall={time.time()-t0:.0f}s "
              f"src={src} len={len(proof)}", flush=True)
        # write incrementally so a crash still yields a partial submission
        write_output(args.output_csv, results)

    trace_f.close()
    await agent.aclose()
    print(f"[run] wrote {len(results)} predictions -> {args.output_csv}", flush=True)


def write_output(path: str, results: list[tuple[str, str]]) -> None:
    p = Path(path)
    if p.suffix != ".csv":           # FMI sometimes passes an output DIR
        p.mkdir(parents=True, exist_ok=True); p = p / "submission.csv"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "prediction"])
        for pid, proof in results:
            w.writerow([pid, proof])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="target model dir (tokenizer + served name)")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--logdir", default="/tmp/proof_logs")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--budget-s", type=float, default=4200.0, help="wall-clock per problem (70min; Kaggle 1h + 30min grace)")
    ap.add_argument("--deadline", type=float, default=0.0, help="overall epoch-sec deadline (0=off)")
    # pool-loop params (validated config — see rehearsal/RESULTS.md)
    ap.add_argument("--select-reserve", type=float, default=900.0, help="reserve window for the selector vote (15min)")
    ap.add_argument("--init-provers", type=int, default=6)
    ap.add_argument("--verify-k", type=int, default=3)
    ap.add_argument("--refine-inputs", type=int, default=4)
    ap.add_argument("--select-bundle-n", type=int, default=4)
    ap.add_argument("--selectors", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128000)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--est-tps", type=float, default=35.0, help="concurrent tok/s for the wall-clock watchdog")
    ap.add_argument("--call-cap", type=int, default=32000, help="per-call token ceiling (above this model's natural proof length)")
    ap.add_argument("--temperature", type=float, default=1.0, help="prover/refiner temp; verify/select use low temp internally")
    ap.add_argument("--health-wait", type=int, default=1200, help="seconds to wait for server")
    # FMI-required knobs (submissions-instructions §2a). Aliases onto the native knobs above so
    # behavior is unchanged when unset.
    ap.add_argument("--top-p", "--top_p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="FMI alias for --max-tokens/--call-cap: per-call generation ceiling")
    ap.add_argument("--num_ctx", type=int, default=None,
                    help="FMI knob: context bound; clamps the per-call ceiling here. The server "
                         "context length itself is set at serve time (serve/serve_final.sh CTX)")
    ap.add_argument("--top_k", type=int, default=None,
                    help="FMI knob: top-k sampling (unset = server default, i.e. disabled)")
    args = ap.parse_args()
    if args.max_new_tokens is not None:
        args.max_tokens = args.max_new_tokens
        args.call_cap = min(args.call_cap, args.max_new_tokens)
    if args.num_ctx is not None:
        args.max_tokens = min(args.max_tokens, args.num_ctx)
        args.call_cap = min(args.call_cap, args.num_ctx)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
