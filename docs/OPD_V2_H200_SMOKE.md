# OPD v2 — single-node 8×H200 end-to-end shakeout (complete step-by-step)

The final integration test before the 64× B200 run: prove the **whole loop** works on one node —
rollout → teacher hidden-extract → full-vocab JSD → FSDP2 step → weight sync → checkpoint — using the
exact B200 code path (`olmo3_sink_fa2` trainer sink + `triton` rollout backend) **and the same producer
as production** (agentic self-play pool, seeded from `ycchen/dsflash-proof-distill-v2-test`), scaled to
fit 8 GPUs. Deliberately minimal: **~57k seq-len (the agentic floor), small batch, ~20 steps.**
*(For a lighter plumbing check without the dataset, flip to `single_round` — see the note at the bottom
of `env_1node_smoke.sh`; it uses the in-repo `problems.parquet` and drops to 40k.)*

Layout (**4:2:2**): teacher DeepSeek-V4-Flash TP4 → GPU 0-3 · rollout student TP2 → 4-5 · trainer
FSDP2+CPU-offload world 2 → 6-7 · orchestrator on CPU.

**How you run these:** on these single-node instances you **SSH straight into the container** (your
prompt is `root@…:/opt/opd/repo/…`), so every command below runs **directly inside the container** — no
`docker run` wrapper, no `-e`/`-v` flags. The whole sequence:

> **preflight → verify image → download models → Shot 1 (plumbing) → watch → Shot 2 (real path)**

The first run is the first time any of this executes on a GPU, so we do it in two shots (§Step 3): a
cheap `single_round` plumbing check, then the real `agentic`+dsflash path.

*(Driving Docker from a separate host instead? Wrap the Step-3 env in
`docker run --rm -it --gpus all --ipc=host --shm-size=64g -v /models:/models -v /runs:/runs -e STUDENT_PATH=… … chankhavu/ycchen-opd:cu128 bash -lc '…'`.)*

---

## Step 0 — preflight (2 min): can the node actually run this?

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv        # expect 8× H200, ~140 GB each
docker run --rm --gpus all chankhavu/ycchen-opd:cu128 nvidia-smi -L # docker sees all 8 GPUs
free -g | awk '/Mem/{print "host RAM:",$2,"GB — need ~400+ (32B + CPU-offloaded optimizer)"}'
df -h /data 2>/dev/null || df -h /                                  # need ~1 TB (DeepSeek + student + scratch)
```
If GPUs aren't visible in docker, host RAM < ~400 GB, or disk < ~1 TB → **stop**; those are hardware
gaps, no config fixes them. (The `docker run … nvidia-smi` also doubles as your image-pull test.)

## Step 1 — get the image (and verify the bring-up fixes are in it)

```bash
docker pull chankhavu/ycchen-opd:cu128
# current digest: sha256:002c8c078393e87c102b3281882aad6ad93ad6949cbb0e963d590b6910fd5ad7
docker image inspect chankhavu/ycchen-opd:cu128 --format '{{index .RepoDigests 0}}'   # must match ^
# air-gapped node instead? on the dev box:  docker save ycchen-opd:cu128 | gzip | ssh h200 'gunzip | docker load'
```

**Verify the two real-hardware fixes are present** (a stale/cached image without them reproduces the
early crashes — CUDA "driver too old" and a `rope_theta` KeyError):
```bash
docker run --rm chankhavu/ycchen-opd:cu128 bash -lc '
  ls /opt/cuda13-compat/libcuda.so* >/dev/null && echo "OK: CUDA-13 forward-compat lib present"
  for s in run_teacher run_rollout; do grep -q cuda13-compat /opt/opd/opd_serve/$s.sh && echo "OK: $s.sh forward-compat preamble"; done
  SGL=$(/opt/venv/serve/bin/python -c "import sglang,os;print(os.path.dirname(sglang.__file__))")
  grep -q "_rope_params.get" "$SGL/srt/models/olmo2.py" && echo "OK: rope_theta fix"'
```
Why they matter here: this node's driver (CUDA 12.8) can't natively run the CUDA-13 sglang serve venv —
the forward-compat lib bridges it (loaded automatically when the driver is < 13); the `rope_theta` fix
lets the Olmo3 **student** load. Both were found + fixed on a live H200 (see
[OPD_V2_H200_BRINGUP_FIXES.md](OPD_V2_H200_BRINGUP_FIXES.md)).

## Step 2 — paths + download the 3 model dirs

The image already ships `hf` (huggingface_hub 1.23) — download inside a throwaway container, no host
Python needed. Both repos are **public** (no token).

```bash
export MODELS=/data/models RUNS=/data/runs      # weights + run scratch (hidden spool, weights, ckpts)
mkdir -p "$MODELS" "$RUNS"

# student (deploy-format; works for BOTH trainer and rollout)
docker run --rm -v "$MODELS":/models chankhavu/ycchen-opd:cu128 \
  hf download chankhavu/yccchen-olmo3-deploy --local-dir /models/student-deploy

# DeepSeek-V4-Flash teacher (large — hundreds of GB; this is the long pole)
docker run --rm -v "$MODELS":/models chankhavu/ycchen-opd:cu128 \
  hf download deepseek-ai/DeepSeek-V4-Flash --local-dir /models/DeepSeek-V4-Flash

# sanity: both must have config.json + shards
ls "$MODELS/student-deploy"/config.json "$MODELS/student-deploy"/*.safetensors | head
ls "$MODELS/DeepSeek-V4-Flash"/config.json
```

**Datasets:** nothing else to download (OPD is on-policy — the student generates its own rollouts, scored
live). The prompt dataset `ycchen/dsflash-proof-distill-v2-test` (public) is seeded into the agentic pool
at runtime by `opd_v2.agentic.seed` → **Shot 2 needs network** on the node (or pre-seed once, see Step 3).
The `single_round` plumbing check (Shot 1) uses the in-repo `problems.parquet` — no seed, no network.

## Step 3 — run the shakeout (recommended: two shots)

This is the first time any of this runs on a GPU. De-risk in two shots: a **fast plumbing check** with
the lighter `single_round` producer (no seed, no network, 40k) to prove the four-process loop turns —
FA2 sink + teacher hidden-extract + weight-sync all fire on real H200s — **then** the real **agentic +
dsflash** path (the baked default, 57k). Shot 1 catches gross breakage cheaply; only escalate to shot 2
once it's green.

### Shot 1 — plumbing check (`single_round`, ~40k, no dataset)

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=64g \
  -v "$MODELS":/models -v "$RUNS":/runs \
  -e STUDENT_PATH=/models/student-deploy \
  -e DEEPSEEK_V4_FLASH=/models/DeepSeek-V4-Flash \
  -e RUN_DIR=/runs/opd_smoke_plumbing \
  -e MOE_BACKEND=marlin \
  -e PRODUCER=single_round \
  -e CONTEXT_LEN=40960 -e MAX_TRAJ_TOKENS=40960 -e MICRO=40960 -e MAX_NEW_TOKENS=36864 \
  -e TRAIN_BATCH_TRAJS=4 -e ROLLOUT_N=2 -e TARGET_INFLIGHT=8 -e ROLLOUT_MAXRUN=4 \
  -e MAX_STEPS=10 \
  chankhavu/ycchen-opd:cu128 \
  bash -lc 'source /opt/opd/launch/env_1node_smoke.sh && bash /opt/opd/launch/run_1node.sh'
```

Prompts come from the in-repo `problems.parquet` — **no seed, no network**. Green = `train/loss` ↓ and
`onpolicy/weight_version` ↑ over ~10 steps. If this OOMs, fix it here (cheaper) before shot 2.

### Shot 2 — the real path (`agentic` + dsflash, ~57k) — the baked default

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=64g \
  -v "$MODELS":/models -v "$RUNS":/runs \
  -e STUDENT_PATH=/models/student-deploy \
  -e DEEPSEEK_V4_FLASH=/models/DeepSeek-V4-Flash \
  -e RUN_DIR=/runs/opd_1node_smoke \
  -e MOE_BACKEND=marlin \
  -e PRODUCER=agentic -e SEED_SOURCE=ycchen/dsflash-proof-distill-v2-test \
  -e REFINE_BUNDLE_CAP=8000 -e SELECT_BUNDLE_CAP=8000 \
  -e CONTEXT_LEN=57344 -e MAX_TRAJ_TOKENS=57344 -e MICRO=57344 -e MAX_NEW_TOKENS=49152 \
  -e TRAIN_BATCH_TRAJS=4 -e ROLLOUT_N=2 -e TARGET_INFLIGHT=8 -e ROLLOUT_MAXRUN=4 \
  -e MAX_STEPS=20 \
  chankhavu/ycchen-opd:cu128 \
  bash -lc 'source /opt/opd/launch/env_1node_smoke.sh && bash /opt/opd/launch/run_1node.sh'
```

Every `-e` here is already the `env_1node_smoke.sh` **baked default** (shown for visibility) — drop them
and just `source env_1node_smoke.sh && bash run_1node.sh`. **The node needs network** for the agentic
seed, or pre-seed once: `python -m opd_v2.agentic.seed --run-dir /runs/opd_1node_smoke`. If the world-2
trainer OOMs at 57k, see Troubleshooting (`MICRO`↓, `TRAINER_NPROC=3`). W&B is **online by default** —
pass `-e WANDB_API_KEY=…` (or `wandb login` in-container) so metrics stream to the cloud; add
`-e WANDB_MODE=offline` for a no-network run and `wandb sync <run-dir>/wandb/offline-run-…` afterward.

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
- **If it fails:** the launcher aborts and names the offending log. Collect the tail of the failing
  server (`{teacher,rollout,trainer}.log`) **and** `orchestrator.log` from `$RUN_DIR` — that pair is
  enough to diagnose almost anything; cross-check the symptom against Troubleshooting below first.

> **Two-shot recap:** Shot 1 writes to `$RUNS/opd_smoke_plumbing/`, Shot 2 to `$RUNS/opd_1node_smoke/` —
> point the `tail -f` commands at whichever shot is running.

---

## Finalization checklist (fresh instance, baked image)

Run on a clean 8×H200 after `docker pull chankhavu/ycchen-opd:cu128` to certify the **baked** image (no
hot-patching). In-container, cheapest→most-expensive:

1. **Image integrity** — `docker image inspect chankhavu/ycchen-opd:cu128 --format '{{index .RepoDigests 0}}'`
   matches `sha256:002c8c078393e87c102b3281882aad6ad93ad6949cbb0e963d590b6910fd5ad7`. In-container the
   flash_rl loader shows 4 fix markers, and forward-compat lib + `curand.h` + `MEMFRAC:-0.70` are baked
   (see the Step-1 verify block above).
2. **Sink correctness** — `python /opt/opd/test_attention_sink.py --list` then `--group all`. Every
   non-skipped target PASS; skips must show a hardware reason, not an error.
3. **Full loop → step + checkpoint** — `single_round`, `-e CHECKPOINT_EVERY=5 -e MAX_STEPS=5 -e HF_EXPORT=1`
   (needs ~500 GB free on `/runs` for one 32B ckpt). Green = `step=5` rc=0, `onpolicy/weight_version` ticks,
   a ckpt lands in `<run>/checkpoints/step_000005/` (+`hf/`, `latest.json`). MEMFRAC 0.70 / WEIGHT_SYNC_EVERY=1
   / W&B online are baked defaults now.
4. **Checkpoint resume** — relaunch the same `RUN_DIR` with `-e RESUME=1 -e MAX_STEPS=10`: trainer DCP-loads
   `latest.json`, health reports `step=5` (not 0), continues 6→10 with continuous loss.

All four green ⇒ **container finalized** — sink + loop + weight-sync + checkpoint + resume validated on the
baked image. Only the B200 sm_100 hardware pass remains (needs a B200; `MOE_BACKEND=auto` picks the
Blackwell teacher backend).

## Config: this shakeout vs. her V33 production

| knob | shakeout | her V33 (B200) | why smaller here |
|---|---|---|---|
| `PRODUCER` | **`agentic`** | `agentic` | **same producer** — real prove/verify/refine/select loop |
| `SEED_SOURCE` | `dsflash-proof-distill-v2-test` | same | **same prompt dataset** |
| `REFINE`/`SELECT_BUNDLE_CAP` | 8k / 8k | 40k / 50k | pulls the agentic ctx floor down to ~56k |
| `CONTEXT_LEN` / `MAX_TRAJ_TOKENS` | 57344 | 130816 | fit 8 GPUs; each traj un-windowed |
| `MICRO` | 57344 (= max_traj) | 131072 | must be ≥ max_traj |
| `MAX_NEW_TOKENS` | 49152 | 128000 | generation cap |
| `TRAIN_BATCH_TRAJS` | 4 | 64 | small batch |
| `TARGET_INFLIGHT` | 8 | 512 | shallow pipeline |
| `TRAINER_NPROC` | 2 | 24 (world) | one node |
| `MAX_STEPS` | 20 | 100000 | just prove it runs |
| `BETA` `LR` `WEIGHT_SYNC_EVERY` `CHUNK_SIZE` `CPU_OFFLOAD` | **1.0 / 1e-5 / 4 / 2048 / on** | same | kept faithful |

A *scaled-down* run of her exact algorithm and dataset, not a different one. (The only knobs that differ
are scale — context, batch, steps, pipeline depth — plus the reduced bundle caps that make the agentic
context floor fit 8 GPUs.)

## Troubleshooting

| symptom | fix |
|---|---|
| trainer OOM (32B, world 2) | `-e MICRO=24576 -e MAX_TRAJ_TOKENS=24576`, or `-e TRAINER_NPROC=3` + steal a rollout GPU (`-e ROLLOUT_GPUS=4 -e ROLLOUT_TP=1 -e TRAINER_GPUS=5,6,7`), or `-e TRAIN_BATCH_TRAJS=2` |
| FSDP mesh balks at world 2 | try `-e TRAINER_NPROC=4` (needs 4 trainer GPUs → drop teacher to `-e TEACHER_TP=2` only if DeepSeek fits, else keep world 2) |
| teacher never healthy | check `teacher.log`; DeepSeek MoE needs 4 GPUs + JIT warmup; raise `-e HEALTH_TIMEOUT=3600` |
| rollout OOM at 57k | `-e MEMFRAC=0.80 -e ROLLOUT_MAXRUN=2` or lower `CONTEXT_LEN` |
| rollout OOM **at weight-sync** (`update_weights_from_disk` → `.clone()` OOM, then a `per_token_group_quant_8bit` CPU `NotImplementedError` in the teardown) | `MEMFRAC` too high for the **fp8 reload peak** (~18-26GB: old fp8 copy + loader clone + bf16 re-quant). On H200 (140GB) use `-e MEMFRAC=0.70` (her 0.82 is for a 180GB B200); drop to `0.65` if it recurs. The CPU `NotImplementedError` is a *symptom* of the OOM (weights fall back to CPU; the fp8 kernel is CUDA-only), not a separate bug — fixing the OOM clears both. |
| agentic won't start (`min_gen_room`) | ctx floor is `48000 + max(bundle caps)`; keep `MAX_TRAJ_TOKENS ≥ 56k` (or drop the caps further) |
| no network for the seed | pre-seed on a connected box: `python -m opd_v2.agentic.seed --run-dir $RUN`, or use `-e PRODUCER=single_round` (in-repo problems) |
| download gated / 401 | accept the DeepSeek license on HF, pass `-e HF_TOKEN=hf_…` to the download container |
| host (CPU) OOM | 32B + offloaded optimizer needs ~400 GB host RAM — no software fix, needs the RAM |

## Next: B200

This smoke already runs the production **agentic** path on the dsflash dataset, so once it's green the
only step left is scale: 64× B200 with `env_v33_b200.sh` + `run_mn_cu128.sh` — her exact V33 (full
130816 ctx, 40k/50k bundle caps, batch 64), only the two B200 deltas. Remaining B200-specific item: a
Blackwell MoE backend for the teacher instead of `marlin` (`MOE_BACKEND=auto` picks it). Launcher
details: [../docker/cu128/launch/README.md](../docker/cu128/launch/README.md).
