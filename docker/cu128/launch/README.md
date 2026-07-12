# docker/cu128/launch — OPD v2 launchers for the cu128 container

Two launchers, both faithful ports of her `training/opd_v2/examples/run_mn.sh` orchestration (same
4-process loop, launch order, health gate). Baked into the image at `/opt/opd/launch/`.

| file | what |
|---|---|
| `run_mn_cu128.sh` | **multi-node** faithful slurm port (her V33 topology). Run inside a slurm allocation. |
| `env_v33_b200.sh` | her V33 production config + the two cu128 deltas. Source before `run_mn_cu128.sh`. |
| `run_1node.sh` | **single-node 8-GPU** integration launcher (co-located, no srun). |
| `env_1node_smoke.sh` | reduced 8-GPU (4:2:2) integration config. Source before `run_1node.sh`. |

The only differences from her scripts are the cu128 container substitutions: `apptainer exec $SIF`
→ our baked `/opt/opd/opd_serve/{run_teacher,run_rollout}.sh` (image `/opt/venv/serve` sglang 0.5.14,
rollout `--attention-backend triton`), her `$REPO/.venv` python → the image train python
(`/opt/conda`, transformers 5.9), and `ATTN_IMPL=olmo3_sink_fa2` (B200 has no FA3).

## Multi-node (64× B200 = 8 nodes, her V33)

```bash
# inside a >=8-node slurm allocation (Beaker-wrapped), with the shared FS mounted:
export STUDENT_PATH=/weka/.../stage1-v2-32b-softdistill-v2test-deploy
export ROLLOUT_MODEL=$STUDENT_PATH STUDENT_DEPLOY_PATH=$STUDENT_PATH
export DEEPSEEK_V4_FLASH=/weka/.../DeepSeek-V4-Flash
export RUN_DIR=/weka/.../runs/opd_v2_$SLURM_JOB_ID
# export MOE_BACKEND=<blackwell backend>    # marlin is Hopper-only; pick one for sm_100
source /opt/opd/launch/env_v33_b200.sh
bash   /opt/opd/launch/run_mn_cu128.sh
```
Topology: 1 teacher (TP4×2) + 4 rollout (TP4×2 fp8 = 8 replicas) + 3 trainer (world 24). Everything
tunable via env (`TEACHER_NNODES`, `ROLLOUT_NNODES`, `ROLLOUT_TP`, …).

## Single-node integration test (8× H200, 4:2:2)

```bash
export STUDENT_PATH=/scratch/.../student-deploy   # deploy-format; works for trainer + rollout
export DEEPSEEK_V4_FLASH=/scratch/.../DeepSeek-V4-Flash
export RUN_DIR=/scratch/.../runs/opd_1node_smoke
source /opt/opd/launch/env_1node_smoke.sh
bash   /opt/opd/launch/run_1node.sh
```
Layout: teacher DeepSeek TP4 [GPU 0-3] · rollout TP2 [4-5] · trainer world 2 [6-7] · orchestrator (CPU).
Default is a **single_round** smoke (context 24k, `MAX_STEPS=30`) — the smallest self-contained loop
(prompts from the in-repo `problems.parquet`). Flip to the **agentic** path (closest to production) per
the note at the bottom of `env_1node_smoke.sh` — that needs context ≥ ~56k (agentic `min_gen_room` floor).

## What "done" looks like (watch the WANDB / stdout metrics)

Loop is healthy when: `train/loss` decreases, `onpolicy/weight_version` climbs (weight sync works),
`perf/rollout_starved_frac` is low (trainer isn't waiting on rollouts), `rollout/length_rate` stays
near 0 (little window truncation), and on `g4/*` steps `g4/top1` rises + `learn/reverse_kl` falls
(student converging to the teacher). See [../../docs/OPD_V2_ALGORITHM.md](../../docs/OPD_V2_ALGORITHM.md).
