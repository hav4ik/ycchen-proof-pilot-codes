# Proof-Pilot

Proof-Pilot is an open-source pipeline for **olympiad-style mathematical proof generation**, built
for the AIMO *Proof Pilot* competition. The task: given 6 olympiad problems, produce a complete
natural-language proof for each, running inside a Kaggle notebook (one RTX 6000 Pro GPU, ~1 hour per
problem). Only whitelisted open models are allowed, so everything here is built on **OLMo 3**.

The delivered system is a **single quantized 32B model** that plays four roles — *prover*, *verifier*,
*refiner*, *selector* — in an agentic loop, driven purely by different prompts. That one model is the
end of a full train-and-distill pipeline whose code lives in this repository.

---

## The delivered model

```
Kaggle submission (kaggle/notebook/proof_pilot_submit_int4mlp.ipynb)
  ├─ target : OPD-32B (step-200) → deploy (GQA-8 / YaRN-256k) → GPTQ-w4a16 → served as W4A8
  │            ↑ OPD v2  — agentic semi-on-policy distillation      (training/opd_v2)
  │            ↑ soft distillation 32B (forward-KL)                 (training/soft_distill_v2)
  │            ↑ SFT stage-1 (L4 data, DeepSeek-teacher mix)        (training/stage1_v2)
  │            ↑ olmo3_sink 32B fused base (learnable attn sink)    (olmo3_sink)
  │            ↑ OLMo 3.1 32B Think + DeepSeek-V4 tokenizer graft   (tokenizer_transplant)
  └─ draft  : DFlash 32B speculative-decoding draft → int4-MLP quant (training/dflash + deploy/dflash)
```

**Teacher / donor:** DeepSeek-V4-Flash (with V4-Pro used for some higher-quality data).
**Evaluation (IMO-ProofBench v2, agentic loop):** the same OPD-32B final proofs score **4.48 / 7**
under an independent Claude grader with SymPy/brute-force checking and **3.808 / 7** under the
calibrated DeepSeek-V4-Flash `high_notool` two-pass grader. Under that same Flash grader, the
DeepSeek-V4-Flash teacher scores **4.83 / 7** (see `evaluation/`).

---

## Pipeline & repository map

| Stage | Directory | What it is |
|---|---|---|
| Tokenizer transplant | `tokenizer_transplant/` | Training-free OMP transplant of the DeepSeek-V4 tokenizer onto OLMo 3 (shared 129k vocab — the distillation linchpin). |
| Custom model | `olmo3_sink/` | OLMo 3 subclass with a learnable gpt-oss-style attention sink, FlashAttention packing-metadata reuse, and a patched in-kernel-sink FA3 backend. |
| Shared training core | `train_core/` | Tokenize + assistant-only loss-mask (`l3_render.py`) and the DeepSeek-V4 chat encoder (`encoding_dsv4.py`). |
| Data generation | `distill_gen/` | DeepSeek-teacher data generation. `math_3r/` is the multi-agent prove→verify→rank→refine→select pipeline; its prompts are the blueprint for the Kaggle inference loop. |
| Data pipeline | `scripts/` | JSONL→Parquet conversion, dataset mixing, verification tooling. |
| SFT | `training/stage1_v2/` | Supervised fine-tuning that produces `stage1-v2-{7b,32b}` (FSDP2/HSDP, L4 pre-packed data). |
| Teacher extraction | `training/teacher_extract/` | Patches sglang to emit DeepSeek teacher hidden states; `dataprep/` renders manifests and extracts the hidden pool for distillation. |
| Soft distillation | `training/soft_distill_v2/` | Offline SFT-shaped distillation (whole-proof rows, chunked fused-linear JSD, forward-KL). |
| On-policy distillation | `training/opd_v2/` | The delivery model. A 4-process online distiller (rollout ‖ teacher ‖ trainer-as-service ‖ orchestrator) running an agentic semi-on-policy pool loop. |
| Speculative decoding | `training/dflash/` | DFlash draft-model training for inference acceleration. |
| Quantization | `quantization/` | llm-compressor GPTQ-w4a16 (the delivered format) + calibration/ablation. |
| Deployment | `deploy/` | sglang serving: `target/` (in-engine sink parity), `quant/` (quantized serving), `dflash/` (speculative decoding), `w4a8/` (W4A8 "humming" runtime patch), `sm120/` (RTX-6000/Blackwell serving patches). |
| Kaggle delivery | `kaggle/` | The submission master: `proof_agent/` (the agentic inference system), `notebook/` (submission notebooks), `serve/`, `bundle/`. |
| Evaluation | `evaluation/`, `evaluation_local/` | IMO-ProofBench harness: single-round + agentic generation, grader calibrated to the human 0/1/6/7 scale, result tables. |

Two shared helpers live under `training/`: `_common/` (the JSD loss kernel + hidden codec) and
`_vendor_opd/` (a small vendored support package the OPD trainer imports).

---

## Three components shared across the whole pipeline

1. **One β-parameterized JSD kernel** (`training/_common/jsd_kernel.py`) — soft distillation uses
   β=0 (forward-KL), OPD uses β=1 (reverse-KL), the refine/select ablation uses β∈(0,1) (JSD). One
   loss implementation spans three training stages.
2. **One set of prompts** — `distill_gen/math_3r`'s prove/verify/refine/select/fallback prompts are
   byte-identical to `kaggle/proof_agent/prompts/`. The training-data generator and the inference
   loop share the same prompt distribution.
3. **One runaway/loop detector** (`zlib_runaway_detector`) — reused in evaluation, in OPD's rollout
   filtering, and in the Kaggle inference loop.

---

## Getting started

```bash
uv sync                                    # Python 3.13; see pyproject.toml
uv run python -c "import torch, transformers, datasets; print('ok')"
```

`olmo3_sink`'s in-kernel-sink backend additionally needs a patched FlashAttention-3 built from source
(see `olmo3_sink/` and `training/stage1_v2/fa3/`); without it, the code falls back to an eager backend.

**Resource locations are read from environment variables** (with neutral defaults), so no absolute
machine paths are baked in. The main ones:

| Variable | Meaning |
|---|---|
| `PROOF_PILOT_ROOT` / `PP_ROOT` | Repository root (most scripts also derive it from `__file__`). |
| `DEEPSEEK_V4_FLASH`, `DEEPSEEK_V4_PRO` | Teacher/donor model directories. |
| `DEEPSEEK_API_KEY` | API key for teacher data generation and grading (never hard-coded). |
| `SGLANG_SIF` | sglang container image for serving/extraction. |
| `STUDENT_PATH`, `TARGET_MODEL_PATH` | Student / target checkpoint directories. |

Each subsystem has its own `README.md` with the concrete commands for that stage.

Large artifacts (model weights, datasets, rollouts, logs) are **not** included — they are gitignored
and regenerable from the code. Training was done on multi-node H200 clusters; the Kaggle path targets
a single RTX 6000 Pro.

---

## License

Apache-2.0 — see [LICENSE](LICENSE). Source files carry Apache-2.0 headers.

Training data is derived from public datasets (Nemotron math/proof collections; DeepSeek-teacher
generations) and released mixes are published under their respective licenses.
