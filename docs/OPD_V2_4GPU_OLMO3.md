# OPD v2 — single-node 4×H200 test: Olmo3-32B ← Olmo3-32B

A smaller integration test than the 8-GPU one, using an **Olmo3-32B teacher** instead of DeepSeek-V4-Flash
(which needs TP4 = all 4 GPUs). Teacher and student are both Olmo3-32B — point them at **two different
checkpoints** for a real distillation (non-zero JSD), or the **same** checkpoint for self-distill
(JSD ≈ 0, which is a strong *correctness* check of the hidden-extract → `W_rot` codec pipeline).

Layout (**1:1:2**): teacher Olmo3-32B TP1 → GPU 0 · rollout Olmo3-32B TP1 → GPU 1 · trainer FSDP2+offload
world 2 → GPU 2-3 · orchestrator CPU.

## Why this works (and why DeepSeek-Tiny doesn't)

- Both models are Olmo3-32B: hidden **5120** (÷32, codec-clean), vocab **129280** (shared) — the teacher
  `/score` hidden feeds the `had+int6_blk32` codec + `W_rot` cleanly.
- The teacher is served by **`run_teacher_olmo3.sh`** (triton sink + `--enable-return-hidden-states` +
  bf16), *not* the DeepSeek `run_teacher.sh`. `patch_dsv4` isn't needed (Olmo3 returns the plain post-norm
  hidden); the 4 model-agnostic plumbing patches (already baked) do the spool/`/score`.
- Two additive config knobs enable it (both default to her exact values): `OPD_HID_DIM=5120` (teacher
  hidden dim; default 4096 = DeepSeek) and `TEACHER_PATH` (the W_rot head source; default = her DeepSeek path).
- ⚠️ `silence09/DeepSeek-V4-Pro-Tiny` **cannot** be used: its hidden=500 fails the codec's hard
  `d % 32 == 0` assert (`_common/hidden_codec.py:52`).

## Download the model(s)

No DeepSeek here — teacher and student are Olmo3-32B. Download the student (public), and a **second**
checkpoint only if you want real distillation (else it self-distills):

```bash
export MODELS=/data/models; mkdir -p "$MODELS"
# student (checkpoint B) — public, no token
docker run --rm -v "$MODELS":/models chankhavu/ycchen-opd:cu128 \
  hf download chankhavu/yccchen-olmo3-deploy --local-dir /models/olmo3-32b-ckptB
# teacher (checkpoint A) — OPTIONAL; a different Olmo3-32B ckpt for a real (non-zero) JSD signal
# docker run --rm -v "$MODELS":/models chankhavu/ycchen-opd:cu128 \
#   hf download <your/olmo3-32b-ckptA> --local-dir /models/olmo3-32b-ckptA
```

**Datasets:** none to download — OPD self-generates its rollouts; the single_round prompts are the
in-repo `distill_gen/problems/problems.parquet` (9,834 problems, ships in the image).

## Run

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=64g \
  -v /data/models:/models -v /data/runs:/runs \
  -e STUDENT_PATH=/models/olmo3-32b-ckptB \
  -e TEACHER_MODEL=/models/olmo3-32b-ckptA \
  -e RUN_DIR=/runs/opd_4gpu_olmo3 \
  chankhavu/ycchen-opd:cu128 \
  bash -lc 'source /opt/opd/launch/env_4gpu_olmo3.sh && bash /opt/opd/launch/run_1node.sh'
```

- **Two checkpoints** → set `TEACHER_MODEL` (checkpoint A) different from `STUDENT_PATH` (checkpoint B):
  real distillation, non-zero JSD.
- **One checkpoint** → drop `-e TEACHER_MODEL=…`: it defaults to `STUDENT_PATH` → self-distill, JSD ≈ 0.
- The preset sets `OPD_HID_DIM=5120`, `TEACHER_PATH=$TEACHER_MODEL`, `TEACHER_SCRIPT=run_teacher_olmo3.sh`,
  the 1:1:2 GPU split, 24k context, batch 4, 20 steps, `CHECKPOINT_EVERY=0`. Override any with `-e`.

## What "healthy" looks like

- **self-distill:** `train/loss` ≈ 0 and stays low — a materially non-zero loss reveals a bug in the
  teacher-hidden / codec / W_rot reconstruction.
- **two checkpoints:** `train/loss` non-zero and (if you let it run) trending down as the student moves
  toward the teacher; `onpolicy/weight_version` climbing; `perf/rollout_starved_frac` low.
- Metric reference: [OPD_V2_ALGORITHM.md](OPD_V2_ALGORITHM.md). Logs in `RUN_DIR/{teacher,rollout,trainer,orchestrator}.log`.

## Fit / troubleshooting

All-32B on 4 GPUs is tighter than the 8-GPU test (teacher bf16 ~64 GB on 1 H200; rollout fp8 ~32 GB;
trainer 32B world-2 + CPU offload). If it's tight:
- trainer OOM → lower `-e MICRO=16384 -e MAX_TRAJ_TOKENS=16384` or `-e TRAIN_BATCH_TRAJS=2`.
- teacher OOM at 28k → lower `-e TEACHER_MEMFRAC=0.80 -e TEACHER_CONTEXT_LEN=20480`.
- first-run check: confirm `run_teacher_olmo3.sh` actually surfaces hidden states (the one unverified
  assumption — that the baked `olmo2.py` returns hidden via `--enable-return-hidden-states`). If `/score`
  returns empty hidden, that's the thing to fix (olmo2.py capture-hidden hook), not the config.
