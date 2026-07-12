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

## Step 0 — image + preflight (you run *inside* the container on these instances)

Pull the image with all the H200 bring-up fixes (CUDA-13 forward-compat + `rope_theta`), then verify:
```bash
docker pull chankhavu/ycchen-opd:cu128
# @ sha256:5c36d05b045426c6356cc7925f9fe4a556d1011a3877660b953c11bdd893773c
```
```bash
ls /opt/cuda13-compat/libcuda.so* && grep -c cuda13-compat /opt/opd/opd_serve/run_teacher_olmo3.sh  # forward-compat present (expect 1)
nvidia-smi -L                                                                                         # 4× H200
```

## Step 1 — download the two checkpoints (real distillation)

- **Student (being trained)** = `opd-32b-v33-s150` — the OPD-final model. It lives in a **subfolder** of
  the deploy bundle, so use `--include`.
- **Teacher (frozen, provides hidden)** = `chankhavu/yccchen-olmo3-deploy` — the pre-OPD deploy student.

Both public, no token, both olmo3 hidden-5120. ~130 GB total.
```bash
export MODELS=/models
# student: opd-32b-v33-s150 (subfolder of the bundle repo)
hf download ycchen/proof-pilot-deploy-bundle --include "opd-32b-v33-s150/*" --local-dir /models/opd-s150
# teacher: the deploy student
hf download chankhavu/yccchen-olmo3-deploy --local-dir /models/teacher-deploy
# sanity
ls /models/opd-s150/opd-32b-v33-s150/config.json /models/teacher-deploy/config.json
```
*(For a **self-distill** codec check instead — JSD ≈ 0 — download only one model and point both roles at it.)*

**Datasets:** none — OPD self-generates its rollouts; single_round prompts ship in-repo
(`distill_gen/problems/problems.parquet`). This test is `single_round` by necessity (agentic's ~56k
floor won't fit a 32B trainer on 2 GPUs); it's a codec / hidden-extract correctness check,
problem-set-independent. The dsflash dataset path is validated by the 8-GPU smoke instead.

## Step 2 — run (student = s150, teacher = deploy)

You're already inside the container, so run the launcher directly (no `docker run` wrapper):
```bash
STUDENT_PATH=/models/opd-s150/opd-32b-v33-s150 \
TEACHER_MODEL=/models/teacher-deploy \
RUN_DIR=/runs/opd_4gpu_distill \
bash -lc 'source /opt/opd/launch/env_4gpu_olmo3.sh && bash /opt/opd/launch/run_1node.sh'
```
The preset supplies the rest: `TEACHER_PATH=$TEACHER_MODEL`, `OPD_HID_DIM=5120`, `run_teacher_olmo3.sh`,
layout **teacher[0] · rollout[1] · trainer[2,3]**, single_round 24k. Two different checkpoints →
**non-zero JSD** (real distillation).

*(From a host with Docker instead of in-container: wrap the same env in
`docker run --rm -it --gpus all --ipc=host --shm-size=64g -v /data/models:/models -v /data/runs:/runs -e STUDENT_PATH=… -e TEACHER_MODEL=… -e RUN_DIR=… chankhavu/ycchen-opd:cu128 bash -lc '…'`.)*

## Watch

```bash
grep -m1 cuda13-compat /runs/opd_4gpu_distill/teacher.log     # forward-compat fired on the teacher
tail -f /runs/opd_4gpu_distill/orchestrator.log              # per-step metrics
```
Olmo3-32B bf16 has no MoE cold-start, so the health gate clears in ~2–5 min (vs 10–20 for DeepSeek).

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
- **hidden export** — verified *in code* (`olmo2.py:501-511` returns the post-norm hidden into
  `logits_processor` with no `before_norm`, so `--enable-return-hidden-states` yields the correct
  `[seq,5120]`; the spool/`/score` patches are model-agnostic and baked), but **not yet GPU-proven for
  Olmo3**. The run itself is the proof: self-distill `train/loss ≈ 0`, or a sane trending-down loss on two
  checkpoints. If `/score` returns *empty* hidden, the fix is the olmo2.py capture-hidden path, not config.
