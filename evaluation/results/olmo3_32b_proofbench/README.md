# OLMo 3 32B — IMO-ProofBench v2 evaluation (agentic prove→verify→refine→select)

Proofs and scores produced by OLMo 3 32B (`olmo3_sink`, the step_200 checkpoint of OPD on-policy
distillation) on **all 60 problems of IMO-ProofBench v2**, using the prove→verify→refine→select
agentic loop. Each problem's final proof is in [`proofs/`](proofs/); the per-problem scores are in
[`SCORES.md`](SCORES.md) / [`scores.tsv`](scores.tsv).

> This directory records two grading views of the **same 60 final proofs**: an independent
> **Claude grader with SymPy/brute-force checking** and the calibrated **DeepSeek-V4-Flash
> `high_notool` two-pass grader**. Scores from different grader families are not directly
> comparable; teacher comparisons below are made within the same grader column.

## Results

### Independent Claude grader

| Split | n | Mean (/7) | solved (≥6) |
|---|---|---|---|
| **Overall** | 60 | **4.48** | 36/60 (60%) |
| Basic | 30 | **6.13** | 26/30 |
| Advanced | 30 | **2.83** | 10/30 |

By difficulty (a strong difficulty gradient):

| level | n | mean | solved |
|---|---|---|---|
| pre-IMO | 8 | **7.00** | 8/8 |
| IMO-easy | 24 | **5.92** | 20/24 |
| IMO-medium | 18 | **3.22** | 7/18 |
| IMO-hard | 10 | **1.30** | 1/10 |

**The score distribution is extremely bimodal**: `7×32, 6×4, 1×21, 0×3`. That is, "either fully correct
(7) or stuck on a core lemma / wrong answer (0–1)", with very little in between. The three 0s are cases
where the answer itself is wrong (PB-Advanced-018 L=2≠3, PB-Advanced-023 3001≠3, PB-Advanced-027
Bob≠Alice).

**Typical failure mode (medium/hard)**: the model correctly simplifies, sets up the framework, and even
guesses the right final answer, but at **the single most critical step** (infinite descent, a key
identity, exhaustive case analysis, an involution lemma, …) it switches to hand-waving or states a
**false lemma refutable by counterexample**, which the grader catches with sympy/brute-force → 1 point.
On easy/pre-IMO problems it almost always produces a complete solution (full marks on pre-IMO).


### DeepSeek-V4-Flash grader

The same final proofs were also scored by `deepseek-v4-flash` using the repository's calibrated
`high_notool` ranking configuration, Appendix B.5 rubric, and two passes per proof:

| Split | n | Mean (/7) |
|---|---:|---:|
| **Overall** | 60 | **3.808** |
| Basic | 30 | **5.600** |
| Advanced | 30 | **2.017** |

The almost-correct-or-better rate is **31/60 (51.7%)**, the fully-correct rate is **27/60 (45.0%)**,
and the two passes agree exactly on **53/60 (88.3%)** proofs. See
[`FLASH_GRADER.md`](FLASH_GRADER.md) for the full breakdown and
[`flash_scores.tsv`](flash_scores.tsv) for both pass scores on every problem.

## Method

- **Model**: OLMo 3 32B = `olmo3_sink` (OLMo 3 + learnable attention sink); the student comes from the
  OPD (on-policy distillation from the DeepSeek-V4-Flash teacher) `agentic_32b_lc140k_v33` run,
  **step_200** checkpoint.
- **Deployment**: training-format checkpoint → `deploy/make_olmo3sink_deploy.py` (legacy-rope/bf16) →
  `kaggle/serve/enable_swa_config.py` (hybrid-SWA) → SGLang **TP4 / fp8 weight / fp8_e4m3 KV / SWA /
  FA3 / reasoning-parser deepseek-r1** (4×H200, `.sif` + `deploy/target/olmo2_sink.py` bind-mount).
- **agentic loop**: `distill_gen/math_3r`'s `solve_problem`, per problem **6 provers → 2 verifiers per
  valid proof → 3 refiners → 4 selectors (majority vote)**, `max_tokens=128000`, `temperature=1.0`. The
  selector only outputs an ID; the proof is retrieved deterministically via a map (no rewriting, so the
  final proof is never truncated).
- **Generation scale**: 60 problems, ~5.8 hours, ~36.5M completion tokens total. The full per-call
  reasoning trace (reasoning_content + content of every prove/verify/refine/select call) is kept locally
  (118MB, not in git; could be released separately on HF).

## Scoring

- 10 Claude grading agents (6 problems each) score with the **IMO 0/1/6/7 rubric** and **actually run
  sympy / brute-force via Bash** to check the final answer and key steps (identities, candidate
  solutions, counterexample search), rigorously catching hand-waving and false lemmas.
- rubric: 0 = almost no progress, 1 = partial progress but an essential gap, 6 = nearly complete with
  only minor blemishes, 7 = fully correct.

### Teacher comparison (apples-to-apples, same grader family)

The teacher (DeepSeek-V4-Flash/Pro) runs **the same math_3r agentic pipeline** and is scored by the
**Claude blind cross-check** from `../agentic_proofbench.md` (`evaluation/harness/claude_xcheck.py`:
Claude sub-agents, B.5 rubric, 0/1/6/7, with numerical verification) — the **same method** as this
evaluation, so it is directly comparable:

| System | Method | Claude grader | flash grader |
|---|---|---|---|
| **OLMo 3 32B (OPD student, s200)** | agentic select | **4.48** | **3.808** |
| DeepSeek-V4-Flash (teacher) | agentic select | **5.30** | 4.83 |
| DeepSeek-V4-Pro | agentic select | 5.32 | 5.31 |

Under the same pipeline and grader, the comparable gaps are:

- Claude grader: student **4.48** vs Flash teacher **5.30** → **−0.82**.
- Flash grader: student **3.808** vs Flash teacher **4.830** → **−1.022**.

The gap is concentrated in medium/hard; on easy/pre-IMO the student is already near the ceiling
(pre-IMO 7.0 under both graders).

Caveats: (1) the two Claude gradings are **different batches** (possible cross-batch variance); (2) the
teacher run was a random pro/flash A/B **blind** grading, whereas this evaluation is not blind; (3) the
`claude_xcheck.py` conclusion also notes that the Claude grader tends to be lenient toward flash's rigor
gaps while the flash grader tends to be strict about catching circular reasoning — **no single grader is
ground truth**, so the most robust approach combines rigor auditing + numerical verification. Single-round
flash baseline (flash grader): best-of-4 4.64, t3 self-verify 4.58.

## Other findings

- **The agentic loop is very stable**: of the 60 problems, **59/60 run a full select**, with only 1
  fallback (all refiners failed). Call-level truncation is 53/1418 (3.7%), but the redundancy of 6
  provers / 3 refiners absorbs it.
- **refine truncation = degenerate loop, not "thinking too long"**: the truncated refiners are cases
  where the reasoning falls into a repetition attractor and runs all the way to the 128k cap (zlib≈0.08,
  the same line repeated 80–214 times, 15-gram 60% repetition); this is the OPD step_200 length
  self-amplification manifesting on the thinking side (details in the repo memories `opd-loop-rootcause`
  / `soft-distill-v2-loops-eos-undertraining`). The verify/refine **inputs contain only the `<solution>`
  body, not the thinking**.

## Reproduction

```bash
# 1) deploy the weights (OPD step_200 -> serve-ready)
python deploy/make_olmo3sink_deploy.py \
  --src training/opd_v2/runs/agentic_32b_lc140k_v33/checkpoints/step_000200/hf \
  --dst outputs/agentic_32b_lc140k_v33-s200-deploy
python kaggle/serve/enable_swa_config.py outputs/agentic_32b_lc140k_v33-s200-deploy
# 2) SGLang TP4 fp8 serve (GPU 0-3): see tmp/pb_agentloop/serve_tp4.sh
# 3) run the agentic loop (math_3r.solve_problem, 6/2/3/4, 128k, temp 1.0) -> save all stages
# 4) Grade the same responses with:
#    (a) Claude agents using 0/1/6/7 + SymPy/brute-force verification
#    (b) grade_proofs.py using deepseek-v4-flash, high_notool, --passes 2
```

## Files

- [`SCORES.md`](SCORES.md) — per-problem Claude-grader score table (links to each proof)
- [`scores.tsv`](scores.tsv) — machine-readable Claude-grader scores
- [`FLASH_GRADER.md`](FLASH_GRADER.md) — Flash grader configuration, aggregate results, and interpretation
- [`flash_scores.tsv`](flash_scores.tsv) — per-problem Flash pass-0/pass-1 scores and their means
- [`proofs/PB-*.md`](proofs/) — for each of the 60 problems: the problem + the model's final proof + score + grader note
