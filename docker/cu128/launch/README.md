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
tunable via env (`TEACHER_NNODES`, `ROLLOUT_NNODES`, `ROLLOUT_TP`, …). Node roles are derived from
`scontrol show hostnames $SLURM_JOB_NODELIST` (teacher = node[0], rollout = nodes[1:5], trainer =
nodes[5:8], rdzv head = node[5]) — not hardcoded.

### Cluster requirements & first-run caveats (audited)

- **Scheduler: this launcher is 100% `srun`-based** (`SLURM_JOB_ID`, `SLURM_JOB_NODELIST`, `scontrol`,
  and per-role `srun --overlap --nodelist …`). It runs on a **slurm** allocation only. If your B200
  Beaker is *not* slurm-backed, the fan-out must be re-ported (torchrun/ray/Beaker primitives) — there
  is no non-slurm path. **Confirm this first.**
- **Shared filesystem** (same mount + path on all 8 nodes, e.g. Weka): `RUN_DIR` is the single source of
  truth — `config.json`, `trainer_endpoint.json`, the teacher-written **hidden-state** files, the
  weight-sync **weights** buffer, DCP **checkpoints**, the agentic **pool/seed.jsonl**, and all logs.
  Model dirs must be readable on the relevant nodes (simplest: also on the shared FS).
- **Node-local, must NOT be shared:** the teacher hidden **spool** `/dev/shm/opd-v2-tea-$port` and
  `TRITON_CACHE_DIR=/tmp/triton_opdv2_$SLURM_NODEID` (per-node by design).
- **Required env (user-supplied — the preset does NOT set these):** `STUDENT_PATH`,
  `STUDENT_DEPLOY_PATH`, `ROLLOUT_MODEL`, `DEEPSEEK_V4_FLASH`, `RUN_DIR`. `DEEPSEEK_V4_FLASH` has a silent
  fallback to `/models/DeepSeek-V4-Flash` — set it explicitly or the teacher fails ~20 min into cold start.
- **Ports** (base + `PORT_SHIFT`, +i/replica): teacher HTTP `8100+` / dist `38100+` / NCCL `38600+`,
  rollout HTTP `8200+`, trainer rdzv `29500+` + HTTP `8300+`. Teacher/rollout HTTP must be reachable from
  the head node (health gate); trainer rdzv+NCCL fabric across the 3 trainer nodes. Set `NCCL_IB_HCA` /
  `NCCL_SOCKET_IFNAME` for your fabric.
- **First-run seeding:** the orchestrator auto-seeds the pool from the public `SEED_SOURCE` at startup
  (no `HF_TOKEN` needed), so it's not a hard prereq — but the **orchestrator node needs network egress**
  and will spend startup time pulling+parsing ~80k records. To skip that, pre-seed once on a connected
  box: `python -m opd_v2.agentic.seed --run-dir $RUN_DIR`.
- **B200 teacher MoE:** `MOE_BACKEND=auto` picks per hardware; marlin is Hopper-only. For the **first**
  B200 run consider pinning `export MOE_BACKEND=flashinfer_mxfp4` (validated for DSv4-Flash TP4 on GB200,
  sglang #23743) rather than trusting auto's first pick after a 20-min cold start.

## Single-node integration test (8× H200, 4:2:2)

```bash
export STUDENT_PATH=/scratch/.../student-deploy   # deploy-format; works for trainer + rollout
export DEEPSEEK_V4_FLASH=/scratch/.../DeepSeek-V4-Flash
export RUN_DIR=/scratch/.../runs/opd_1node_smoke
source /opt/opd/launch/env_1node_smoke.sh
bash   /opt/opd/launch/run_1node.sh
```
Layout: teacher DeepSeek TP4 [GPU 0-3] · rollout TP2 [4-5] · trainer world 2 [6-7] · orchestrator (CPU).
Default is the **agentic** path (57k, the same producer + dsflash dataset as production, `MAX_STEPS=20`)
so the smoke exercises the real data path. For a lighter first plumbing check, flip to `single_round`
(40k, in-repo `problems.parquet`, no seed/network) — see the two-shot launch order in
[../../docs/OPD_V2_H200_SMOKE.md](../../docs/OPD_V2_H200_SMOKE.md).

## What "done" looks like (watch the WANDB / stdout metrics)

Loop is healthy when: `train/loss` decreases, `onpolicy/weight_version` climbs (weight sync works),
`perf/rollout_starved_frac` is low (trainer isn't waiting on rollouts), `rollout/length_rate` stays
near 0 (little window truncation), and on `g4/*` steps `g4/top1` rises + `learn/reverse_kl` falls
(student converging to the teacher). See [../../docs/OPD_V2_ALGORITHM.md](../../docs/OPD_V2_ALGORITHM.md).
