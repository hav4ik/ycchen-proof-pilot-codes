# docker/cu128 — OPD v2 on CUDA 12.8 (B200 / sm_100)

Containerizes the **full OPD v2 loop** (FSDP2 trainer + orchestrator + rollout sglang +
DeepSeek-V4-Flash teacher) into one **cu128** image for Blackwell B200. Student =
`chankhavu/yccchen-olmo3-deploy`.

## Why cu128 (not the cu13 production stack)

The Ai2/Beaker cluster ceiling is **CUDA 12.8**, which is also the floor for Blackwell
`sm_100`. B200 has **no FA3** (Hopper-only), so two substitutions vs. the H200 production run:

| component | H200 production (cu13) | this image (cu128 / B200) |
|---|---|---|
| trainer sink | patched-FA3 in-kernel | **`olmo3_sink_fa2`** — post-correction on **stock** flash-attn-2 (no kernel patch) |
| rollout attn | FA3 default (honors sink) | **`--attention-backend triton`** — honors the sink in **prefill *and* decode**; flashinfer drops it |

Numerics are validated fp64-exact vs eager and bit-exact on the OPD JSD loss. The rollout
sglang uses Yi-Chia's patched sources, re-anchored to the **0.5.14** release (her cu13
binaries won't run on cu128). Full parity matrix: [`OPD_V2_PARITY_STATUS.md`](OPD_V2_PARITY_STATUS.md).

## Build

Build context is the **repo root** (the Dockerfile `COPY .`s the tree into the image):

```bash
DOCKER_BUILDKIT=1 docker build -f docker/cu128/Dockerfile.ycchen-opd -t ycchen-opd:cu128 .
```

Base: `chankhavu/olmo3-olmocore:cu128-fa2-sink` (torch 2.10+cu128, flash_attn 2.8.x, olmo_core).
Override with `--build-arg BASE=...`. The build self-verifies both venvs (trainer import +
sink registration; sglang serve patches incl. the hybrid-SWA KV-pool patch; teacher
hidden-extract patches).

## Layout

| path | purpose |
|---|---|
| `Dockerfile.ycchen-opd` | the cu128 image (two venvs: base trainer/orchestrator, `/opt/venv/serve` sglang) |
| `opd_serve/run_teacher.sh` | DeepSeek-V4-Flash `/score` teacher (hidden-state extraction → shared FS) |
| `opd_serve/run_rollout.sh` | student rollout (fp8 + flash_rl loader; `--attention-backend triton`) |
| `opd_serve/sglang_patches/` | Yi-Chia's patched sglang sources (olmo2 sink model, flash_rl loader, load_config, server_args) |
| `opd_v2_train_smoke.py` | fp64-exactness + OPD JSD + FA2-parity + compile smoke (GPU) |
| `test_attention_sink.py` | central sink-test launcher (`--group all`) |
| `OPD_V2_PARITY_STATUS.md` | cu13 ↔ 0.5.14 parity/差異 matrix + adversarial findings |
| `OPD_V2_TRAIN_DOCKER_PLAN.md` | design/plan doc |

The two anchored source patchers live in the repo (not here) and run at build time:
`training/opd_v2/flash_rl/patches/apply_swa_patch.py` (rollout SWA KV-pool) and
`training/teacher_extract/_patch_sglang_514.py` (teacher hidden-extract, 5 patches).

## Launch (single node, one GPU per role)

```bash
# teacher (DeepSeek-V4-Flash)
CUDA_VISIBLE_DEVICES=3 /opt/opd/opd_serve/run_teacher.sh --tp 1 --port 8100
# rollout (student, deploy-format checkpoint)
CUDA_VISIBLE_DEVICES=1 MODEL=<student-deploy> /opt/opd/opd_serve/run_rollout.sh --port 8201
# trainer
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m opd_v2.trainer.service --run-dir $OPD_RUN_DIR
# orchestrator
ATTN_IMPL=olmo3_sink_fa2 ROLLOUT_URLS=... TEACHER_URLS=... python examples/make_config.py \
  && python -m opd_v2.orchestrator --run-dir $OPD_RUN_DIR
```

Multi-node is a mechanical adaptation of `training/opd_v2/examples/run_mn.sh` (swap
`apptainer exec $SIF` → these baked venvs, slurm → Beaker).
