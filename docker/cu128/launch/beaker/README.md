# OPD v2 on Beaker — V33 64x B200 (DRAFT)

Beaker launcher for Yi-Chia Chen's **V33** OPD v2 run: **64x B200 = 8 nodes x 8 GPU**, packaged in
`chankhavu/ycchen-opd:cu128`. This is the Slurm→Beaker port of the verified slurm launcher.

> **Status:** the specs are **pre-filled against the Ai2 Beaker docs** for **`ai2/titan-cirrascale`** (96× B200,
> 192 GB) — cluster, NCCL (`ib`/`^=mlx5_bond_0`), `gpuCount`, `sharedMemory`, `timeout`, and the pinned image are
> set. You fill **3 team-specific values** (budget, Weka bucket+`subPath`, priority). Run the 3-node smoke first,
> then the 8-node run. **Full prerequisites → [AI2_HANDOFF.md](AI2_HANDOFF.md).**

## Prerequisites (before submitting) — full detail in [AI2_HANDOFF.md](AI2_HANDOFF.md)
- **Models** (download to Weka, mount read-only): teacher `deepseek-ai/DeepSeek-V4-Flash` → `/models/DeepSeek-V4-Flash`
  (~83 GB); student `chankhavu/yccchen-olmo3-deploy` → `/models/student-deploy` (~64 GB; one deploy checkpoint
  serves BOTH trainer + rollout).
- **Seed dataset**: auto-fetched at runtime into `<RUN_DIR>/pool/seed.jsonl` (public mirror
  `chankhavu/ycchen-dsflash-proof-distill-v2-test`) — nothing to download on a networked node. For an offline
  cluster, pre-build once: `python -m opd_v2.agentic.seed --run-dir <RUN_DIR>`.
- **Storage**: a WRITABLE shared Weka bucket for **`RUN_DIR`** (the loop's single source of truth) + a **FIXED**
  shared **`JIT_CACHE_DIR`** (persists the one-time ~10-20 min DeepGEMM + ~15 min fp4-autotune warm across runs).

| file | what | topology |
|---|---|---|
| [`opd_v33_b200.yaml`](opd_v33_b200.yaml) | **production** — 8×B200, her V33 knobs, `replicas: 8` | 1+4+3 |
| [`opd_smoke3_b200.yaml`](opd_smoke3_b200.yaml) | **launcher smoke** — 24×B200, 57k ctx, 20 steps, `replicas: 3` | 1+1+1 |
| [`run_mn_beaker.sh`](../run_mn_beaker.sh) | the launcher each replica runs; role dispatch by `BEAKER_REPLICA_RANK`. | — |

It is a **faithful** port of [`../run_mn_cu128.sh`](../run_mn_cu128.sh): identical role scripts,
port scheme (`PORT_SHIFT`), health gate, launch order, `make_config`, `opd_v2.trainer.service`
(c10d torchrun), orchestrator, and NCCL env. **Only Slurm→Beaker node-discovery/placement changes.**

## Submit

```bash
# 1) (recommended) import the docker image into Beaker for faster pulls, then set image.beaker:
#    beaker image create chankhavu/ycchen-opd:cu128 --name ycchen-opd-cu128
# 2) fill the PLACEHOLDERs, then:
beaker experiment create docker/cu128/launch/beaker/opd_v33_b200.yaml
```

Watch: `beaker experiment logs <id>` (per-replica). Health/metrics land in `wandb` and in
`$RUN_DIR/launch_rank*.log`, `$RUN_DIR/{teacher,rollout,trainer}_*.log`, `$RUN_DIR/orchestrator.log`.

## Replica rank → role

Each of the 8 replicas runs the **same** command; `run_mn_beaker.sh` picks its role from
`BEAKER_REPLICA_RANK` (rank ordering == the slurm node ordering). Defaults from `env_v33_b200.sh`:
`TEACHER_NNODES=1`, `ROLLOUT_NNODES=4` → `TRAINER_NNODES = 8 − 1 − 4 = 3`.

| rank | role | what runs on the node | GPUs |
|:---:|---|---|:---:|
| 0 | **teacher** | `run_teacher.sh` x `TEACHERS_PER_NODE=2`, TP4 (DeepSeek-V4-Flash) | 8 |
| 1–4 | **rollout** | `run_rollout.sh` x `ROLLOUTS_PER_NODE=2`, TP4 fp8 (student, triton sink) | 8 |
| 5 | **trainer + orchestrator** | torchrun (FSDP2) node 0 = **rdzv head**; also health gate → `make_config` → **orchestrator** (CPU) | 8 |
| 6–7 | **trainer** | torchrun (FSDP2) — join the c10d rendezvous at rank-5's hostname | 8 |

Trainer world size = `8 × TRAINER_NNODES = 24`. Note the **rdzv head is rank 5**, *not* the Beaker
leader (rank 0); the head's hostname is discovered from the gather (below), not from
`BEAKER_LEADER_REPLICA_HOSTNAME`.

## How node discovery works (the one thing that changed) — READ THIS

Slurm gave the head an **ordered node list** via `scontrol show hostnames $SLURM_JOB_NODELIST`, and
one process fanned work to every node with `srun --overlap --nodelist=<n>`. Beaker has **neither**:
it starts 8 independent copies of the script and injects into each only its own `BEAKER_REPLICA_RANK`
and (leader only) `BEAKER_LEADER_REPLICA_HOSTNAME`. **Non-leader replica hostnames are not injected.**

So the launcher rebuilds the ordered node list on the **shared FS**:

1. Every replica writes its own routable hostname — `BEAKER_NODE_HOSTNAME` (Beaker injects this into
   *every* job) — to `$RUN_DIR/.beaker_hosts_<id>/rank_<rank>`.
2. Every replica barriers until all `BEAKER_REPLICA_COUNT` files exist, then reads them back in rank
   order → `NODES[]`. This array is identical in meaning to the slurm one, so **all downstream
   topology math, ports, and URLs are byte-for-byte hers**.

Why `BEAKER_NODE_HOSTNAME`: for the leader it equals `BEAKER_LEADER_REPLICA_HOSTNAME` (the docs'
routable rdzv endpoint), so the same source is routable for non-leaders under `hostNetworking`. The
launcher **cross-checks** this on rank 0 and warns if they differ. The **health gate is the empirical
proof**: if a teacher/rollout hostname were non-routable, rank 5 could not curl `…/health` and the
gate would time out before any training — a fast, safe failure.

**Residual risk (flagged):** if `BEAKER_NODE_HOSTNAME` on the target B200 cluster is *not* reachable
peer-to-peer (unlikely, but not verified here), the teacher/rollout endpoint URLs and the trainer
rdzv endpoint would be wrong. Mitigations if the rank-0 canary warns or the health gate times out:
set `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` correctly first; if hostnames still don't resolve, override the
per-replica name (e.g. add a small step that resolves the routable name and export it as
`BEAKER_NODE_HOSTNAME`), or ask #beaker-users for the cluster's peer-routable hostname/IP source.

## What was preserved vs adapted

**Preserved verbatim from `run_mn_cu128.sh`** (do not re-tune): role scripts
`/opt/opd/opd_serve/run_{teacher,rollout}.sh` and their per-role env; `PORT_SHIFT` and the
teacher/rollout/trainer port scheme; the health gate (same curl, 1800s timeout, 10s poll); launch
**order** (teacher+rollout → health → `make_config` → trainer → orchestrator); `make_config.py` with
`ATTN_IMPL=olmo3_sink_fa2`; the c10d torchrun `opd_v2.trainer.service`; the orchestrator invocation;
`NCCL_ENV`; `TRITON_CACHE_DIR` per node; the `pkill` cleanup patterns.

**Adapted (forced by Slurm→Beaker only):**
- **node list**: `scontrol` → shared-FS rank→hostname gather barrier.
- **HOLDER**: numeric `$SLURM_JOB_ID` → a hash of `BEAKER_EXPERIMENT_ID` (numeric, identical on all
  replicas) for `PORT_SHIFT` / rdzv-id / tag. Slurm's job id was numeric; no Beaker id is.
- **placement**: `srun --overlap --nodelist=$n <body>` → `<body>` run locally on the matching replica.
- **trainer**: one cross-node `srun torchrun` → each trainer replica runs its **own** torchrun; they
  c10d-rendezvous into one world at rank-5's hostname (we keep her c10d rendezvous, so node-rank is
  assigned by the rendezvous — we do **not** pass `--node-rank`, matching her script).
- **ordering across replicas**: non-head trainers wait for `$RUN_DIR/config.json` before torchrun
  (reproduces "make_config before trainer"); the head removes stale `trainer_endpoint.json`/`STOP`.
- **cleanup**: srun-fanned `pkill` → local `pkill` per replica + a `$RUN_DIR/STOP` sentinel the head
  writes on exit so servers/other trainers shut down (the slurm EXIT trap did this implicitly).
  Crash teardown is also handled by `propagateFailure`/`propagatePreemption`.
- **logs**: one head `launch.log` → per-replica `launch_rank<rank>.log` (avoids clobbering).

## Shared filesystem requirement (CRITICAL)

`RUN_DIR` is the **single source of truth** shared by all 8 replicas: `config.json`,
`trainer_endpoint.json`, the teacher hidden-state spool index, the weight-sync buffer, rolling
weights, DCP checkpoints, the agentic pool, the rank→hostname gather, and all logs. It **must be one
writable filesystem mounted at the same path on every replica** (Weka).

> The Beaker DataMount docs say datasets mount **read-only**, but **WEKA experiment mounts are
> read-write in practice at Ai2** (that is how checkpoints get written from batch jobs). **Confirm the
> run-dir mount is writable on your target cluster before launch** — the entire loop coordinates
> through `RUN_DIR`, so a read-only mount fails immediately at the first write (the gather).

Node-local (must **not** be shared, and are not): the teacher spool `/dev/shm/opd-v2-tea-$port`
(needs `resources.sharedMemory` ≫ the 5GiB default — see the placeholder) and
`TRITON_CACHE_DIR=/tmp/triton_opdv2_$TRAINER_LOCAL_RANK`.

## PLACEHOLDER checklist (fill before submit)

In `opd_v33_b200.yaml` (values are quoted so the file stays valid YAML — replace the whole quoted string):

- [ ] `budget` — your Beaker budget (e.g. `ai2/oe-training`).
- [ ] `constraints.cluster` — the **B200 cluster** name.
- [ ] `context.priority` — `low`/`normal`/`high`/`urgent`.
- [ ] `image` — keep `docker: chankhavu/ycchen-opd:cu128`, or import it and set `image.beaker`.
- [ ] `datasets[].source.weka` + `subPath` — the **writable** Weka bucket/prefix for the run dir,
      and the (read) model dirs for `DeepSeek-V4-Flash` and the student deploy dir. Mount paths must
      not overlap.
- [ ] `RUN_DIR` env — a path under the **writable** `/weka/run` mount.
- [ ] `DEEPSEEK_V4_FLASH`, `STUDENT_PATH`, `ROLLOUT_MODEL`, `STUDENT_DEPLOY_PATH` env — the mount
      paths. `STUDENT_DEPLOY_PATH` is used by weight-sync saves; if it needs to be **written**, point
      it at a writable location (under `/weka/run`), not a read-only model mount. Verify against
      `run_agentic_mn_32b.sbatch` / her config before the full run.
- [ ] `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` env — the **B200 fabric** IB HCA / socket ifname. The
      beaker-docs suggest `NCCL_SOCKET_IFNAME=ib`, `NCCL_IB_HCA=^=mlx5_bond_0` for ai2 IB clusters —
      confirm for the target cluster (ask #beaker-users if unsure).
- [ ] `resources.sharedMemory` — big enough for the teacher `/dev/shm` spool (default 5GiB is too small).
- [ ] `timeout` — max job lifespan (cluster may cap it).
- [ ] *(optional)* `MOE_BACKEND=flashinfer_mxfp4` — marlin is Hopper-only; pin a Blackwell backend
      for the first B200 run (sglang #23743). `WANDB_API_KEY`/`HF_TOKEN` via **secrets** (literal
      names are blocked); `MAX_STEPS` override.

## Smoke first

Do **not** launch 8 nodes cold. Shrink to 3 nodes (1 teacher + 1 rollout + 1 trainer) — the launcher
computes `TRAINER_NNODES` from whatever it gets, so only three envs + `replicas` change:

```yaml
replicas: 3
# envVars:
#   - {name: TEACHER_NNODES, value: "1"}
#   - {name: ROLLOUT_NNODES, value: "1"}   # -> TRAINER_NNODES = 3 - 1 - 1 = 1 (world 8)
#   - {name: MAX_STEPS,      value: "20"}
```

This exercises the exact code path that matters here — the **hostname gather + health gate +
cross-node c10d rendezvous** — at 1/3 the cost. If the gather barrier completes, the health gate
passes (proving peer hostname routability), and the world-8 trainer rendezvous forms, the discovery
mechanism is sound and the only remaining scale-up variable is capacity. Then set `replicas: 8` and
restore `ROLLOUT_NNODES=4` (or just drop the smoke overrides to fall back to the `env_v33_b200.sh`
defaults) for the full run.
