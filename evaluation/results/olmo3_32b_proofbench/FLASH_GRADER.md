# DeepSeek-V4-Flash grading — OLMo 3 32B OPD step 200

This records a second grading view of the **same 60 final proofs** in this directory. The proofs were
generated once by the OLMo 3 32B `olmo3_sink` checkpoint from the
`agentic_32b_lc140k_v33` OPD run at **step 200**, using the
prove → verify → refine → select agentic loop.

## Grader configuration

- grader model: `deepseek-v4-flash`
- configuration: `high_notool`
- prompt: IMO-ProofBench Appendix B.5 rubric
- scale: `0 / 1 / 6 / 7`
- passes: **2 per proof** (120 grader calls total)
- aggregation: each proof's score is the mean of its two pass scores
- exact pass agreement: **53/60 (88.3%)**
- empty grader outputs: **0**

## Results

| Split | n | Mean (/7) |
|---|---:|---:|
| **Overall** | 60 | **3.808** |
| Basic | 30 | **5.600** |
| Advanced | 30 | **2.017** |

| Difficulty | n | Mean (/7) |
|---|---:|---:|
| pre-IMO | 8 | **7.000** |
| IMO-easy | 24 | **5.625** |
| IMO-medium | 18 | **2.000** |
| IMO-hard | 10 | **0.150** |

| Category | n | Mean (/7) |
|---|---:|---:|
| Algebra | 16 | **4.438** |
| Combinatorics | 16 | **3.938** |
| Number theory | 14 | **4.286** |
| Geometry | 14 | **2.464** |

- almost-correct-or-better (`score >= 6`): **31/60 (51.7%)**
- fully correct (`score == 7`): **27/60 (45.0%)**

Per-problem pass scores and their means are in [`flash_scores.tsv`](flash_scores.tsv).

## Same-grader comparison

All systems below use the same agentic evaluation setup and the same calibrated Flash grader:

| System | Score (/7) |
|---|---:|
| **OLMo 3 32B OPD student, step 200** | **3.808** |
| DeepSeek-V4-Flash teacher | **4.830** |
| DeepSeek-V4-Pro teacher | **5.310** |

The student trails its Flash teacher by **1.022 points** under this grader.

## Interpreting the two grader views

The same student proofs score **4.48/7** under the independent Claude grader with
SymPy/brute-force checking and **3.808/7** under the calibrated DeepSeek-V4-Flash grader. This is
grader sensitivity on a fixed set of proofs, not a difference between model checkpoints or generation
runs. Comparisons to teacher systems should only be made within the same grader column.
