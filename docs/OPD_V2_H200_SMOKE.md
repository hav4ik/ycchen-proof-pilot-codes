# OPD v2 — single-node 8×H200 end-to-end shakeout (complete step-by-step)

The final integration test before the 64× B200 run: prove the **whole loop** works on one node —
rollout → teacher hidden-extract → full-vocab JSD → FSDP2 step → weight sync → checkpoint — using the
exact B200 code path (`olmo3_sink_fa2` trainer sink + `triton` rollout backend), scaled to fit 8 GPUs.
Deliberately minimal: **40k max seq-len, small batch, ~20 steps.**

Layout (**4:2:2**): teacher DeepSeek-V4-Flash TP4 → GPU 0-3 · rollout student TP2 → 4-5 · trainer
FSDP2+CPU-offload world 2 → 6-7 · orchestrator on CPU.

Every command below is copy-paste, top to bottom. Run them **on the H200 node** unless marked *(dev box)*.

---

## Step 0 — pick your paths

```bash
export MODELS=/data/models          # where model weights live
export RUNS=/data/runs              # scratch for run outputs (hidden spool, weights, config, ckpts)
mkdir -p "$MODELS" "$RUNS"
# the DeepSeek teacher repo you served in production — CONFIRM this repo id (likely deepseek-ai/DeepSeek-V4-Flash):
export DEEPSEEK_REPO=deepseek-ai/DeepSeek-V4-Flash
# if the teacher repo is gated, accept its license on HF and set a token:
# export HF_TOKEN=hf_xxx
```

Disk: the student is ~65 GB; **DeepSeek-V4-Flash is large (hundreds of GB)** — make sure `$MODELS` has room.

## Step 1 — download the models

The image already has `hf` (huggingface_hub 1.23), so run these inside a throwaway container that mounts
`$MODELS` (no local Python setup needed). Or run `hf download …` directly if you have the HF CLI on the host.

```bash
# 1a. student (deploy-format; PUBLIC; works for BOTH trainer and rollout)
docker run --rm -v "$MODELS":/models chankhavu/ycchen-opd:cu128 \
  hf download chankhavu/yccchen-olmo3-deploy --local-dir /models/student-deploy

# 1b. DeepSeek-V4-Flash teacher  (set HF_TOKEN if the repo is gated)
docker run --rm -e HF_TOKEN -v "$MODELS":/models chankhavu/ycchen-opd:cu128 \
  hf download "$DEEPSEEK_REPO" --local-dir /models/DeepSeek-V4-Flash
```

Sanity-check both have a `config.json` + weight shards:
```bash
ls "$MODELS/student-deploy"/config.json "$MODELS/student-deploy"/*.safetensors | head
ls "$MODELS/DeepSeek-V4-Flash"/config.json | head
```

## Step 2 — get the image onto the node

Built on the dev box; ship it to the H200 node one of two ways:

```bash
# (dev box) push to a registry you own:
docker tag ycchen-opd:cu128 chankhavu/ycchen-opd:cu128 && docker push chankhavu/ycchen-opd:cu128
# (H200 node) pull:
docker pull chankhavu/ycchen-opd:cu128

# --- or transfer directly, no registry: ---
# (dev box)   docker save ycchen-opd:cu128 | gzip | ssh h200 'gunzip | docker load'
```

## Step 3 — run the shakeout

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=64g \
  -v "$MODELS":/models -v "$RUNS":/runs \
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

The `-e` overrides are optional — they are the `env_1node_smoke.sh` defaults, shown explicitly so the
config is visible in one place. Drop them for the baked defaults; change one to retune (e.g.
`-e MICRO=24576`). For live metrics add `-e WANDB_MODE=online -e WANDB_API_KEY=…`.

## Step 4 — watch it (from another shell on the node)

```bash
tail -f "$RUNS/opd_1node_smoke/launch.log"          # orchestration progress
tail -f "$RUNS/opd_1node_smoke/orchestrator.log"    # per-step metrics
# server startup:  tail -f "$RUNS/opd_1node_smoke/"{teacher,rollout}.log
# trainer:         tail -f "$RUNS/opd_1node_smoke/trainer.log"
```

- **Startup is slow:** DeepSeek cold-start + JIT can take ~10-20 min; the health gate (`HEALTH_TIMEOUT`
  1800s) blocks the trainer until both servers answer `/health`. If a server dies early the launcher
  aborts and points at the log.
- **Healthy loop:** `train/loss` ↓, `onpolicy/weight_version` ↑ (sync works), `perf/rollout_starved_frac`
  low, `rollout/length_rate` ≈ 0; on g4 steps `g4/top1` ↑ + `learn/reverse_kl` ↓. Metric reference:
  [OPD_V2_ALGORITHM.md](OPD_V2_ALGORITHM.md).
- **Done:** orchestrator exits rc=0 after `MAX_STEPS`; a checkpoint under `$RUNS/opd_1node_smoke/checkpoints/`.

---

## Config: this shakeout vs. her V33 production

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

A *scaled-down* run of her exact algorithm, not a different one.

## Troubleshooting

| symptom | fix |
|---|---|
| trainer OOM (32B, world 2) | `-e MICRO=24576 -e MAX_TRAJ_TOKENS=24576`, or `-e TRAINER_NPROC=3` + steal a rollout GPU (`-e ROLLOUT_GPUS=4 -e ROLLOUT_TP=1 -e TRAINER_GPUS=5,6,7`), or `-e TRAIN_BATCH_TRAJS=2` |
| FSDP mesh balks at world 2 | try `-e TRAINER_NPROC=4` (needs 4 trainer GPUs → drop teacher to `-e TEACHER_TP=2` only if DeepSeek fits, else keep world 2) |
| teacher never healthy | check `teacher.log`; DeepSeek MoE needs 4 GPUs + JIT warmup; raise `-e HEALTH_TIMEOUT=3600` |
| rollout OOM at 40k | `-e MEMFRAC=0.80 -e ROLLOUT_MAXRUN=2` or lower `CONTEXT_LEN` |
| download gated / 401 | accept the DeepSeek license on HF, pass `-e HF_TOKEN=hf_…` to the download container |
| host (CPU) OOM | 32B + offloaded optimizer needs ~400 GB host RAM — no software fix, needs the RAM |

## Next: agentic path, then B200

Once single_round is green, flip to the **agentic** producer (the real production path) at the ~57k
floor — the note at the bottom of `env_1node_smoke.sh`. Then scale to 64× B200 with `env_v33_b200.sh`
+ `run_mn_cu128.sh` (only remaining B200-specific item: a Blackwell MoE backend for the teacher instead
of `marlin`). Launcher details: [../docker/cu128/launch/README.md](../docker/cu128/launch/README.md).
