# evaluation — IMO-ProofBench harness

Proof-quality evaluation for the OPD-32B system, built on **IMO-ProofBench** (olympiad proofs
graded on the human `0 / 1 / 6 / 7` scale). Used to set the teacher ceiling, calibrate an
automatic grader against human scores, sweep prover prompts, and score the delivered model.

## Two generation modes
- **Single-round** — one prover call per problem (`run_eval.py`; `run_template_sweep.py` sweeps
  8 prompt templates). Produces `runs/<id>/responses.jsonl`.
- **Agentic** — the delivered pipeline: `prove → verify → refine → select` (`run_agentic_eval.py`,
  reusing `distill_gen/math_3r`). `agentic_to_responses.py` flattens a full trace into the same
  `responses.jsonl` schema so both modes share the grader.

## Grading
`grade_proofs.py` scores proofs with a DeepSeek grader **calibrated to human scores**
(`high_notool`, paper B.5 prompt, 2 passes; ~0.87 aggregate Pearson — see
`results/grader_calibration.md`). Proofs are meta-stripped to `graded_text` (`extract_proof.py`)
so a model's own self-assessment can't leak into grading. `analyze_sweep.py` / `score.py` aggregate.

## Layout
| path | purpose |
|---|---|
| `harness/run_eval.py` | single-round candidate generation (OpenAI-compatible endpoint) |
| `harness/run_agentic_eval.py`, `tool_loop.py` | agentic prove/verify/refine/select generation |
| `harness/grade_proofs.py`, `grader.py`, `calibrate_grader.py` | calibrated grader + calibration |
| `harness/extract_proof.py` | meta-stripper → `graded_text` |
| `harness/analyze_sweep.py`, `score.py` | aggregation / ranking |
| `harness/claude_xcheck.py` | independent blind cross-grading |
| `prompts/` | prover + grader (paper B.5) templates |
| `data/` | ProofBench v2 CSVs + single-round template set (tracked benchmark data) |
| `results/` | grader calibration, k=4 analysis, template sweep, and the OPD-32B run |

## Usage
```bash
export DEEPSEEK_API_KEY=...        # grading needs API access
# 1) generate (against any OpenAI-compatible endpoint)
python harness/run_eval.py --data data/proofbench_v2.csv \
  --base-url http://127.0.0.1:30000/v1 --served-model default --run-id myrun --k 4
# 2) grade
python harness/grade_proofs.py --run runs/myrun
```

## Headline results
- Teacher ceiling: DeepSeek-V4-Flash best-of-4 **4.64/7** under the calibrated Flash grader.
- Teacher agentic single output: **4.83/7** under the Flash grader (beats best-of-4); refine is the
  quality engine (+0.92/proof).
- Delivered OPD-32B, same 60 final proofs:
  - **4.48/7** under the independent Claude grader with SymPy/brute-force checking
    (Basic 6.13 / Advanced 2.83; pre-IMO 7.0 → IMO-hard 1.3).
  - **3.808/7** under the calibrated DeepSeek-V4-Flash `high_notool` two-pass grader
    (Basic 5.600 / Advanced 2.017; fully correct 27/60).
- Same-grader agentic comparison: OPD student **3.808** vs Flash teacher **4.830** vs Pro teacher
  **5.310**. See `results/olmo3_32b_proofbench/FLASH_GRADER.md`.
