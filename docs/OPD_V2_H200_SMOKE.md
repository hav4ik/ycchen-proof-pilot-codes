# OPD v2 — single-node 8×H200 end-to-end shakeout

The final integration test before the 64× B200 run: prove the **whole loop** works on one node —
rollout → teacher hidden-extract → full-vocab JSD → FSDP2 step → weight sync → checkpoint — using the
exact B200 code path (`olmo3_sink_fa2` trainer sink + `triton` rollout backend), just small enough to
fit 8 GPUs. Deliberately minimal: **40k max seq-len, small batch, ~20 steps.**

Layout (**4:2:2**): teacher DeepSeek-V4-Flash TP4 → GPU 0-3 · rollout student TP2 → 4-5 · trainer
FSDP2+CPU-offload world 2 → 6-7 · orchestrator on CPU.

## 0. Prerequisites on the H200 node

- **Image** `ycchen-opd:cu128` present on the node. It's built on the dev box; ship it:
  ```bash
  # dev box:
  docker tag ycchen-opd:cu128 chankhavu/ycchen-opd:cu128 && docker push chankhavu/ycchen-opd:cu128
  # H200 node:
  docker pull chankhavu/ycchen-opd:cu128
  # (or transfer directly:  docker save ycchen-opd:cu128 | gzip | ssh h200 'gunzip | docker load')
  ```
- **Models** on a mounted path (`/data/models/`):
  - student, **deploy-format** (`chankhavu/yccchen-olmo3-deploy` — verified loadable by *both* trainer
    and rollout) → `/data/models/student-deploy`
  - teacher `DeepSeek-V4-Flash` → `/data/models/DeepSeek-V4-Flash`
- **Scratch** dir for `RUN_DIR` (hidden spool, rolling weights, config, checkpoints) → `/data/runs`.
- **Host RAM** ≥ ~400 GB (32B + CPU-offloaded optimizer states). **`--shm-size` ≥ 64g** (teacher hidden spool).
- Hopper needs no MoE-backend change: the teacher's default `marlin` works on H200 (B200 would not).

## 1. Full run command (paste-ready)

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=64g \
  -v /data/models:/models -v /data/runs:/runs \
  -e STUDENT_PATH=/models/student-deploy \
  -e DEEPSEEK_V4_FLASH=/models/DeepSeek-V4-Flash \
  -e RUN_DIR=/runs/opd_1node_smoke \
  -e MOE_BACKEND=marlin \
  -e CONTEXT_LEN=40960 -e MAX_TRAJ_TOKENS=40960 -e MICRO=40960 -e MAX_NEW_TOKENS=36864 \
  -e TRAIN_BATCH_TRAJS=4 -e ROLLOUT_N=2 -e TARGET_INFLIGHT=8 -e ROLLOUT_MAXRUN=4 \
  -e MAX_STEPS=20 \
  chankhavu/ycchen-opd:cu128 \
  bash -lc 'source /opt/opd/launch/env_1node_smoke.sh && bash /opt/opd/launch/run_1node.sh'
```

The `-e` overrides are optional — they're the env_1node_smoke.sh defaults, shown explicitly so the
config is visible in one place. Drop them to use the baked defaults, or change them to retune. For
live metrics, add `-e WANDB_MODE=online -e WANDB_API_KEY=…`.

## 2. What each knob is (this shakeout vs. her V33 production)

| knob | shakeout | her V33 (B200) | why smaller here |
|---|---|---|---|
| `PRODUCER` | `single_round` | `agentic` | no `min_gen_room` floor → smallest ctx; validates the core loop |
| `CONTEXT_LEN` / `MAX_TRAJ_TOKENS` | 40960 | 130816 | fit 8 GPUs; each traj un-windowed |
| `MICRO` | 40960 (= max_traj) | 131072 | must be ≥ max_traj |
| `MAX_NEW_TOKENS` | 36864 | 128000 | generation cap |
| `TRAIN_BATCH_TRAJS` | 4 | 64 | small batch |
| `ROLLOUT_N` | 2 | (n/a agentic) | fewer samples/prompt |
| `TARGET_INFLIGHT` | 8 | 512 | shallow pipeline |
| `TRAINER_NPROC` | 2 | 24 (world) | one node |
| `MAX_STEPS` | 20 | 100000 | just prove it runs |
| `BETA` `LR` `WEIGHT_SYNC_EVERY` `CHUNK_SIZE` `CPU_OFFLOAD` | **1.0 / 1e-5 / 4 / 2048 / on** | same | kept faithful |

Everything below the line stays at her values — this is a *scaled-down* run of her exact algorithm,
not a different one.

## 3. Watching it

Logs land in `RUN_DIR` (`/data/runs/opd_1node_smoke/`): `teacher.log`, `rollout.log`, `trainer.log`,
`orchestrator.log`, `launch.log`.

- **Startup is slow:** DeepSeek cold-start + JIT can take ~10-20 min; the health gate (`HEALTH_TIMEOUT`
  1800s) blocks the trainer until both servers answer `/health`. If a server dies early the launcher
  aborts and points at the log.
- **Healthy loop** (orchestrator stdout / wandb): `train/loss` ↓, `onpolicy/weight_version` ↑ (sync
  works), `perf/rollout_starved_frac` low, `rollout/length_rate` ≈ 0; on g4 steps `g4/top1` ↑ +
  `learn/reverse_kl` ↓. Metric reference: [OPD_V2_ALGORITHM.md](OPD_V2_ALGORITHM.md).
- **Done:** orchestrator exits rc=0 after `MAX_STEPS`; a checkpoint under `RUN_DIR/checkpoints/`.

## 4. Troubleshooting

| symptom | fix |
|---|---|
| trainer OOM (32B, world 2) | lower `MICRO`/`MAX_TRAJ_TOKENS` (e.g. 24576), or `TRAINER_NPROC=3` (steal a rollout GPU: `ROLLOUT_GPUS=4 ROLLOUT_TP=1 TRAINER_GPUS=5,6,7`), or `TRAIN_BATCH_TRAJS=2` |
| FSDP mesh balks at world 2 | `TRAINER_NPROC=4` with `TEACHER_TP=2` (frees 2 GPUs) — note teacher TP2 may not fit DeepSeek; else try world 2 first |
| teacher never healthy | check `teacher.log`; DeepSeek MoE needs 4 GPUs + JIT warmup; raise `HEALTH_TIMEOUT` |
| rollout OOM at 40k | lower `MEMFRAC` (0.80), `ROLLOUT_MAXRUN` (2), or `CONTEXT_LEN` |
| host OOM (CPU) | reduce optimizer footprint: fewer trainer ranks won't help (offload is per-param); needs the RAM |

## 5. Next: the agentic path, then B200

Once the single_round smoke is green, flip to the **agentic** producer (the real production path) at the
~57k floor — see the note at the bottom of `env_1node_smoke.sh`. Then scale to 64× B200 with
`env_v33_b200.sh` + `run_mn_cu128.sh` (only remaining B200-specific item: pick a Blackwell MoE backend
for the teacher instead of `marlin`). Launcher details: [../docker/cu128/launch/README.md](../docker/cu128/launch/README.md).
