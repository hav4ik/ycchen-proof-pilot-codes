# docs/ — OPD pipeline updates (branch `opd/b200-cu128`)

This folder tracks the changes we make to Yi-Chia Chen's OPD v2 pipeline on this branch.
The goal of the branch: run the **full OPD v2 loop on CUDA 12.8 / B200 (sm_100)**, staying
faithful to her cu13 behavior. Her training *code* is unchanged; our work is the cu128
port of the attention-sink kernels, the sglang serve patches, and the container.

## Documents
| doc | what it covers |
|---|---|
| [OPD_V2_ALGORITHM.md](OPD_V2_ALGORITHM.md) | how her algorithm works — 4-process arch, the agentic self-play environment, the rollout `produce_sample` atom, JSD distillation, weight-sync & staleness |
| [OPD_V2_CONFIG_REFERENCE.md](OPD_V2_CONFIG_REFERENCE.md) | every config knob + env override + her best OPD-32B run (V33) values |
| [OPD_V2_PARITY_STATUS.md](OPD_V2_PARITY_STATUS.md) | the cu13 ↔ cu128/sglang-0.5.14 parity matrix, adversarial findings, and what's baked/verified |
| [OPD_V2_TRAIN_DOCKER_PLAN.md](OPD_V2_TRAIN_DOCKER_PLAN.md) | the original design/plan for the cu128 image |
| [../docker/cu128/README.md](../docker/cu128/README.md) | how to build/run the cu128 image |

## Change log (newest first)

### Container launchers (faithful ports of run_mn.sh)
Added `docker/cu128/launch/`: `run_mn_cu128.sh` (multi-node, her V33 topology) + `env_v33_b200.sh`,
and `run_1node.sh` (single-node 8-GPU, 4:2:2 teacher:rollout:trainer) + `env_1node_smoke.sh`. Only the
cu128 substitutions differ from her run_mn.sh (our baked role scripts, image train python,
`ATTN_IMPL=olmo3_sink_fa2`). Baked into the image at `/opt/opd/launch/`.

### Documented her algorithm + config
Added [OPD_V2_ALGORITHM.md](OPD_V2_ALGORITHM.md) (environment + rollout + training loop, read from her
code) and [OPD_V2_CONFIG_REFERENCE.md](OPD_V2_CONFIG_REFERENCE.md) (all knobs + her best OPD-32B V33
run values). These are faithful reference docs — behavior descriptions, no changes.

### Adversarial faithfulness audit (5 sub-agents) — 1 violation found & fixed
Audited the entire branch diff vs her code; the only allowed changes are (1) cu128 packaging
and (2) the FA2 attention-sink port. Results:
- **`register.py` / `make_config.py`** — FAITHFUL (additive/opt-in; `ATTN_IMPL` defaults to her
  `olmo3_sink_fa3`). Fixed a misleading comment that suggested stock `flash_attention_2` (which
  silently drops the sink) — corrected to `olmo3_sink_fa2`.
- **`olmo3_sink_fa2.py`** — FAITHFUL PORT (forward `o_sink`, sink-inclusive `lse'`, exact
  dq/dk/dv via FA2 native backward, identical closed-form `dsink`, varlen isolation, sliding
  window). Added a flash-attn `>=2.7` version guard (the private varlen fwd/bwd are called
  positionally with split `window_size_left/right`).
- **`_patch_sglang_514.py`** — FAITHFUL RE-ANCHOR (all teacher patches map 1:1 to her
  `_patch_sglang.py`; emit-rank, fp8 `wo_a` bf16 dequant, /score, spool all identical; the
  5th patch restores her 0.5.12 no-truncation in spool mode).
- **serve `sglang_patches/*`** — byte-identical to her patched sglang in `pp-env`.
- **`run_teacher.sh` / `run_rollout.sh` / Dockerfile** — FAITHFUL except **one violation, now
  fixed**: `run_rollout.sh` defaulted `SKIP_TOKENIZER_INIT=1` (dropped her parsers/tokenizer);
  restored to her default `0`. (Output-neutral for the token-in/out `/generate` path, but an
  undocumented flag divergence.)

### Faithfulness guard — restored her weight-sync behavior
The working tree carried local edits to `orchestrator.py` / `data_plane/clients.py` /
`trainer/core.py` that **changed** her design (abort-pause + `flush_cache=True` +
fail-closed all-or-nothing barrier + a producer-side sink-checkpoint validator) rather than
only fixing bugs — her code deliberately uses `in_place` + `flush_cache=False` ("under
in_place a failed flush asserts and kills the scheduler") and tolerates partial replica
reloads (semi-on-policy). These were **reverted to her exact `74faacb` code** to stay
faithful; the departed work + its two tests are preserved on branch
**`wip/weight-sync-barrier`** for later review as an explicit opt-in.
(`build_l4.py` still carries one benign SFT data-prep bugfix — drop truncated docs missing
EOS to avoid training non-termination — pending a keep/revert decision.)

### cu128/B200 port
- **Container** `docker/cu128/Dockerfile.ycchen-opd` — full loop, two isolated venvs
  (base trainer/orchestrator; `/opt/venv/serve` rollout + teacher sglang 0.5.14). Builds
  from the repo root; self-verifies both venvs at build time.
- **Trainer sink** `olmo3_sink/olmo3_sink_fa2.py` (new) — B200 has no FA3, and stock
  transformers `flash_attention_2` silently drops the sink, so the sink runs as a
  `torch.compile`-safe **post-correction on stock FA2** (`o·exp(lse−logaddexp(lse,sink))`,
  in-place, zero extra memory). Validated **fp64-exact** vs eager and **bit-exact** on the
  OPD JSD loss; cross-checked identical to Prime-RL and OLMo-core.
  Registered in `olmo3_sink/register.py`; selectable via `ATTN_IMPL` in
  `training/opd_v2/examples/make_config.py`.
- **Rollout attention** — `--attention-backend triton` (not the H200 FA3 default). Verified
  triton threads the sink through **both** `forward_extend` (prefill) and `forward_decode`
  (decode) on 0.5.14, so every decoded token gets the sink (flashinfer drops it).
- **Rollout SWA KV-pool** — the build runs her idempotent
  `training/opd_v2/flash_rl/patches/apply_swa_patch.py` against stock 0.5.14 `model_config.py`
  so `is_hybrid_swa_model(Olmo3Sink)=True`; without it `--swa-full-tokens-ratio` is silently
  inert and the 140k rollout OOMs.
- **Teacher hidden-extract** `training/teacher_extract/_patch_sglang_514.py` (new) — Yi-Chia's
  DeepSeek-V4-Flash `/score` hidden-extraction patches **re-anchored to sglang 0.5.14** (5
  patches; 0.5.14 split her single 0.5.12 output-processor into
  `batch_result_processor.py` + `output_streamer.py`). Fixes two adversarially-found bugs:
  per-rank spool writes via `get_attention_tp_rank()`, and a per-chunk truncation skip in
  spool mode. `wo_a` forced to dequant-to-bf16 (`SGLANG_OPT_FP8_WO_A_GEMM=0`) to avoid
  fp8 policy bias in the teacher hidden state.

## Still open
- Multi-node/Beaker launcher — mechanical adaptation of `training/opd_v2/examples/run_mn.sh`
  (swap `apptainer exec $SIF` → the container's two venvs; slurm → Beaker).
- B200 hardware pass — DeepSeek-V4-Flash MoE backend (`marlin` → sm_100
  cutlass/triton/flashinfer) and an actual sm_100 kernel run.
