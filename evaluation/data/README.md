# Evaluation data — IMO-ProofBench

Source: Google DeepMind *IMO-Bench* (paper *Towards Robust Mathematical Reasoning*, EMNLP 2025, arXiv:2511.01846).

| File | Contents | sha256 |
|---|---|---|
| `proofbench_v2.csv` | **IMO-ProofBench**, 60 proof problems (Basic 30 + Advanced 30). **Use v2** (v1 is deprecated; it fixed the PB-Advanced-022 typo). | `aa8b813dbd4068137e3d165e5da228f6e0e1cc85a91c37883e1791b954e43af0` |
| `gradingbench.csv` | **IMO-GradingBench**, autograder calibration reference (problem × candidate solution × human score). Used to validate the correlation between the ProofAutoGrader and human scores; not the main eval set. | `e85a520c2bbb5a89f2db35088c7935d485922f26cf2bbd04f15e1d967af26cfe` |
| `deepseek-math-v2.txt` | 8,192-token-window math worked-solutions text used as the **attention-sink calibration corpus** by `olmo3_sink/build_init_model.py` (Final Pipeline step 2); its sha256 is recorded in each sink checkpoint's `sink_provenance.json`. | `d3f478ac5917266ae92b061ea0b56480baa8a0753a8b05125873e121b35864c8` |
| `imo-bench.txt` | Plain-text dump of IMO-Bench problems/solutions (auxiliary reference; not the eval set). | `a46c57bb1c5cfecd473a3cc3f0e1e8840bf2e1ae5051e1e054728866350ef14b` |
| `subset_dev.csv` | Small dev subset of ProofBench used for harness smoke tests. | `8d55ae6f4214d837f5780b9ba09852ac3a7ca8388161044d64fb43a5ab24904f` |
| `imo_proofbench_single_round_prompt_templates.json` | Single-round grading prompt templates for the ProofAutoGrader. | `07ec8aa41d40e3fdaaa721930954513e6fdbe2b026bb12d193953bbdaef8844c` |

Downloaded from: `https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/` (fetched 2026-06-01).

License: data is **CC-BY 4.0**, original source code Apache-2.0. Cite arXiv:2511.01846 / imobench.github.io.

## proofbench_v2.csv columns

`Problem ID, Problem, Solution, Grading guidelines, Category, Level, Short Answer, Source`

- **Problem ID**: `PB-Basic-NNN` / `PB-Advanced-NNN`.
- **Problem / Solution**: LaTeX problem statement and reference solution.
- **Grading guidelines**: the grading rubric (states the conditions each `(Partial)` / `(Almost)` level must meet); fed to the grader.
- **Category**: Algebra(16) / Combinatorics(16) / Number theory(14) / Geometry(14).
- **Level**: pre-IMO(8) / IMO-easy(24) / IMO-medium(18) / IMO-hard(10).
- **Short Answer**: the final answer (may be blank for pure proof problems).
- **Source**: origin, e.g. `(Modified) IMO 2019, P1`.

## Contamination note

The Advanced subset is mostly newly written by medalists / robustified adaptations (to prevent memorization); the Basic subset includes adapted public problems. If the SFT data (Nemotron / Cascade-2) contains olympiad sources, check for overlap with this set before evaluating (especially the Basic subset).
