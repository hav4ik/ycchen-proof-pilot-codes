# OPD v2 — Ai2 operator handoff (what YOU need to provide)

Everything else — the training/serving code, both venvs, Yi-Chia's patches, the launchers, the teacher's
`flashinfer_mxfp4` MoE auto-detect — is **baked into the image**. To run the loop on your Beaker cluster you
provide exactly three things: **(1) the 2 model checkpoints in the right mount paths, (2) a WRITABLE shared
run dir, and (3) a WRITABLE shared JIT-compile cache dir.** Then fill the site-specific placeholders and submit.

**Image (ship digest, drift-clean, all B200 fixes validated on B200):**
```
chankhavu/ycchen-opd:cu128@sha256:908516a710f3f6c4157a92c0f9723ff862f84a9c3bf347601c6b2b9ca0ef1a25
```
(Optionally import it into Beaker for faster pulls than Docker Hub, then use the `beaker:` image field.)

**In this folder:** [`README.md`](README.md) (submit steps + launcher internals) · **[`opd_max_b200.yaml`](opd_max_b200.yaml)
— the RECOMMENDED 64×B200 run** (her V33 pipeline + ~25% longer rollout, 160k) · [`opd_v33_b200.yaml`](opd_v33_b200.yaml)
(byte-faithful V33 baseline, 128k — for the exact reproduction) · [`opd_smoke3_b200.yaml`](opd_smoke3_b200.yaml)
(3-node launcher smoke) · [`DEBUGGING.md`](DEBUGGING.md) (which log → which symptom → fix). **This file is the
prerequisites + spec-filling guide** — it applies to all three.

---

## 1 · Download the 2 models to Weka (both are PUBLIC — no HF token)

| role | HF repo | ~size | **mount path in the job** |
|---|---|---|---|
| **teacher** | `deepseek-ai/DeepSeek-V4-Flash` | ~83 GB (fp4 experts + fp8 dense) | `/models/DeepSeek-V4-Flash` |
| **student** | `chankhavu/yccchen-olmo3-deploy` | ~64 GB (deploy-format Olmo3-32B) | `/models/student-deploy` |

**Only ONE student checkpoint is needed** — the deploy-format model works for BOTH the trainer and the rollout
(`STUDENT_PATH` = `ROLLOUT_MODEL` = `STUDENT_DEPLOY_PATH` = `/models/student-deploy`).

Download **once** to a Weka location (the mounts below make it read-only in the job). HF's `xet`/`hf_transfer`
accelerators can stall on large files — disable them for reliability:
```bash
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
hf download deepseek-ai/DeepSeek-V4-Flash  --local-dir /weka/<bucket>/models/DeepSeek-V4-Flash  --max-workers 4
hf download chankhavu/yccchen-olmo3-deploy --local-dir /weka/<bucket>/models/student-deploy      --max-workers 4
# sanity: each dir must have config.json + model-*.safetensors shards
```
The `hf` CLI is inside the image, so you can also run these from a throwaway container.

### Seed dataset — auto-fetched at runtime (NO mount needed); pre-build only for offline
The agentic pool is seeded from **`chankhavu/ycchen-dsflash-proof-distill-v2-test`** (PUBLIC). By default the
orchestrator does a live `load_dataset(...)` at pool init → the **raw** dataset lands in the HF datasets cache
(`~/.cache/huggingface`, transient) and is parsed into **`<RUN_DIR>/pool/seed.jsonl`** (on Weka, persists). It is
**not** a `/models` mount, so on a networked node there is **nothing to pre-download** — it just works.

**To run fully offline / avoid the one-time runtime fetch, pre-build the seed once** (the run then skips it,
because it skips if `<RUN_DIR>/pool/seed.jsonl` exists non-empty):
```bash
SEED_SOURCE=chankhavu/ycchen-dsflash-proof-distill-v2-test \
  python -m opd_v2.agentic.seed --run-dir <RUN_DIR>      # inside the image, on a networked prep node
# -> <RUN_DIR>/pool/seed.jsonl (1776 problems); the job reads it from the shared Weka RUN_DIR
```

### Prerequisites checklist (do these before submitting)
- [ ] **Teacher** `deepseek-ai/DeepSeek-V4-Flash` downloaded to Weka, mounted at `/models/DeepSeek-V4-Flash`.
- [ ] **Student** `chankhavu/yccchen-olmo3-deploy` downloaded to Weka, mounted at `/models/student-deploy`.
- [ ] **`RUN_DIR`** and **`JIT_CACHE_DIR`** are WRITABLE shared Weka paths (JIT_CACHE_DIR fixed, not per-run).
- [ ] JIT cache: **nothing to do** — the baked compile cache **and** fp4 autotune seed into `JIT_CACHE_DIR` on launch (ON by default), so a cold first run reaches `/health` in ~1–2 min (§2). Override with `JIT_CACHE_SEED=0` (seed nothing) or `JIT_AUTOTUNE_SEED=0` (compile only, re-tune per hardware).
- [ ] Seed: nothing (auto-fetched at runtime) — OR pre-build `<RUN_DIR>/pool/seed.jsonl` for an offline cluster.
- [ ] **Every `<PLACEHOLDER: …>` in the yaml replaced** — the infra ones (budget, Weka bucket+subPath, priority, cluster/NCCL) are your call; the rest (RUN_DIR, model paths, JIT_CACHE_DIR) just un-wrap the recommended value. See §3.
- [ ] `WANDB_API_KEY` set as a Beaker secret (optional; W&B is online by default).

## 2 · Provide the three writable/shared paths

| path (in-job) | must be | why |
|---|---|---|
| `/models/DeepSeek-V4-Flash`, `/models/student-deploy` | mounted (read-only OK) | the checkpoints from step 1 |
| **`RUN_DIR`** (e.g. `/weka/run/opd_max`) | **WRITABLE + shared + identical path on every replica; a DISTINCT path per run** | the loop's single source of truth: `config.json`, trainer endpoint, teacher hidden-state spool index, weight-sync buffer, rolling weights, **DCP + HF checkpoints**, the rank→hostname gather, all logs. A read-only mount fails at the first gather. **Each run needs its OWN `RUN_DIR`** — the three yamls default to distinct paths (`/weka/run/opd_max_b200`, `opd_v33_b200`, `opd_smoke3_b200`); never share one, since their `config.json` + agentic pools have different context lengths (160k / 128k / 57k) and would clobber each other. |
| **`JIT_CACHE_DIR`** (e.g. `/weka/jit_cache`) | **WRITABLE + shared + FIXED (not per-run)** | the DeepGEMM/triton/flashinfer JIT-compile cache. The serve stack **writes** compiled kernels here. Make it a **fixed** path (NOT under `RUN_DIR`) so it **persists across runs and instances**. The image seeds its baked compile cache **+ fp4 autotune** into this dir on launch (ON by default — see below), so a cold first run reaches `/health` in ~1–2 min. |

**About the JIT cache (your "writable/saveable cache dir"):** on a fully cold node the DeepSeek-V4 teacher
spends **~10–20 min** JIT-compiling fp8/fp4 kernels per GEMM shape **plus ~15 min** on the flashinfer fp4 MoE
autotune — **~25–35 min before `/health` returns 200** (the profiling forward that triggers the autotune runs
*during* init, before the port opens). The shipped image removes essentially all of it:

- **Both the compile cache AND the fp4 autotune ship BAKED INTO THE IMAGE, seeded ON by default.** It carries a
  pre-built sm_100/TP4 cache at `/opt/opd/jit-cache-seed/sm100/{teacher,rollout}` — compile cubins
  (deep_gemm/triton/flashinfer/tvm-ffi) **and** the flashinfer fp4 autotune (`sglang/flashinfer/autotune/…`).
  On launch `run_{teacher,rollout}.sh` copy it into your `JIT_CACHE_DIR` (`cp -rn`, **never clobbering a warmer
  cache**), so **a cold first run reaches `/health` in ~1–2 min instead of ~25–35 min** — no download, no prep.
- **Two off-switches** (both default on): `JIT_CACHE_SEED=0` seeds nothing (image never touches your cache dir);
  `JIT_AUTOTUNE_SEED=0` keeps the compile cache but drops the baked autotune so it re-tunes for your hardware.
- **The autotune is env-specific but should transfer**: it's keyed by `flashinfer-version + sm100 + shape-hash`
  (all identical across B200 nodes running this pinned image at TP4) — **not** by driver/CUDA version. Worst
  case if it doesn't hit, flashinfer re-tunes (~15 min, still under the launcher's **30 min health-gate**, and
  it then persists in `JIT_CACHE_DIR`). Correctness is never at risk (every cached tactic is a valid kernel).
  If you'd rather each site re-tune from scratch, set `JIT_AUTOTUNE_SEED=0`.

Point `JIT_CACHE_DIR` at Weka so whatever is (re)tuned persists across restarts + replicas. The baked cache is
the public dataset [`chankhavu/opd-jit-cache-sm100`](https://huggingface.co/datasets/chankhavu/opd-jit-cache-sm100)
(`opd-jit-sm100-tp4-autotune.tgz`) — **you do not need to fetch it** (it's in the image). Scope: **sm_100
(B200/B300) at TP4** (matches `TEACHER_TP=ROLLOUT_TP=4`); a different arch/TP just recompiles (harmless).

### Recovery: if the pre-built cache ever misbehaves (clean **and** disable — both are required)
The seed is copied into `JIT_CACHE_DIR` **at startup, by default** — and because `JIT_CACHE_DIR` is a
**persistent** Weka path, once a run has seeded it the copy lives there independently of the flags. So if you
ever suspect the baked cache is behind a kernel error/hang (rare — same arch+version, and every cached tactic
is a valid kernel), you must do **BOTH**, in order:
1. **Delete the persistent cache** so the run rebuilds fresh (a stale/bad entry is already on disk — flipping a
   flag alone will NOT remove it):
   ```bash
   rm -rf "$JIT_CACHE_DIR"/sm100/          # drops the seeded cache AND any compiled/tuned entries; safe — it's a cache
   ```
2. **Turn off seeding** so the next launch doesn't just re-copy the baked cache. Set in the yaml:
   - `JIT_AUTOTUNE_SEED=0` → keep the portable compile cache, only re-tune the fp4 autotune (**try this first** —
     the autotune is the env-specific part, so it's the likely culprit), **or**
   - `JIT_CACHE_SEED=0` → seed nothing; recompile **and** re-tune everything from scratch on your hardware (nuclear).

The first launch after this pays the one-time compile/autotune (~16–35 min), then persists a clean,
hardware-native cache that every later run + replica reuses. Re-enable the seed (drop the override) only after
regenerating the dataset from *your* hardware, if ever.

## 3 · Fill the Beaker spec — most of it is already done

All three yamls (`opd_max_b200.yaml` recommended, `opd_v33_b200.yaml` faithful baseline, `opd_smoke3_b200.yaml`
smoke) share the same placeholders and come with a
**working default** for the mechanical bits; the infra-specific values are yours to set — you know your
environment better than any default we'd guess.

Set already (adjust if your cluster differs):
- **`gpuCount: 8`**, **`sharedMemory: 128GiB`** (the teacher's hidden-state spool lives in `/dev/shm`), `timeout`,
  and the **pinned image digest**.
- **NCCL for multi-node IB**: `NCCL_SOCKET_IFNAME=ib`, `NCCL_IB_HCA=^=mlx5_bond_0`, `NCCL_DEBUG=INFO` — the common
  Ai2 IB values; confirm they match your fabric.
- **Cluster (candidate):** `ai2/titan-cirrascale` (B200) or `ai2/holmes` (B300). Requirement: **sm_100** nodes,
  8 GPU/node — production needs **8 nodes (64 GPU)**, the smoke 3. Our image needs **CUDA 12.8+ / torch ≥2.7**
  (trainer torch 2.10+cu128, serve 2.11+cu130) — satisfied.

**Replace every `<PLACEHOLDER: …>` before submitting.** Most just need un-wrapping — they carry a recommended
value (`RUN_DIR`, the model mount paths, `JIT_CACHE_DIR`, the subPaths). A few are genuinely your call, and we
don't presume your values:
- **`budget`** — your Beaker budget account.
- **`weka: <bucket>` + `subPath:`** — the writable shared storage you use for training state, for the three
  mounts (the run dir + the two model dirs from step 1). `RUN_DIR` / `JIT_CACHE_DIR` must be paths under that mount.
- **`context.priority`** — your call.

W&B is online by default — set `WANDB_API_KEY` from a Beaker **secret** (`beaker secret write wandb-api-key <key>`,
then uncomment the `secret:` line). `HF_TOKEN` is **not** needed (models + seed dataset are public).

## 4 · Submit: smoke first, then the full run

```bash
# (a) 3-node launcher smoke — validates rank->role, the shared-FS hostname gather, cross-node c10d, health gate
beaker experiment create docker/cu128/launch/beaker/opd_smoke3_b200.yaml
#     green = all 3 replicas gather hostnames, teacher+rollout health-pass, trainer forms world 8,
#             steps 1..20 with loss ↓, weight-sync ticking (footer of the yaml has the full checklist)

# (b) full 64x B200 — the RECOMMENDED run: opd_max (her V33 pipeline + ~25% longer rollout, 160k ctx)
beaker experiment create docker/cu128/launch/beaker/opd_max_b200.yaml
#     ! on step 1, confirm the trainer doesn't OOM at MICRO=163840; if it does, back off MICRO +
#       MAX_TRAJ_TOKENS together by ~8k (the yaml header documents this single knob).

# (alt) the byte-faithful V33 baseline (128k) — run this instead only for Yi-Chia's exact reproduction:
# beaker experiment create docker/cu128/launch/beaker/opd_v33_b200.yaml
```
**`opd_max` is the config to run** — it's her exact V33 with only the rollout length raised ~25% (every
training knob identical, in-distribution for the student, memory-safe on B200). **`opd_v33`** is the
byte-faithful 128k baseline, kept for reference / exact reproduction. Give **each run its own `RUN_DIR`**
(`/weka/run/opd_max_b200` vs `.../opd_v33_b200` vs `.../opd_smoke3_b200`) — never shared, since their
`config.json` + agentic pools have different context lengths. They **do** share `JIT_CACHE_DIR` on purpose
(`/weka/run/jit_cache`) — the smoke warms kernels the full run reuses; `cp -rn` is additive.

**If anything misbehaves → [`DEBUGGING.md`](DEBUGGING.md)** — the log map (`orchestrator.log` = the training
heartbeat, `trainer_<K>.log`, `teacher_*`/`rollout_*.log`, `launch_rank<N>.log`) and a symptom → fix table for
every failure mode we hit during bring-up.

---

## What runs (no action needed — for context)

- **Topology (her V33): 1 teacher + 4 rollout + 3 trainer = 8 nodes × 8 B200 = 64 GPUs**, world-24 trainer.
  (`ROLLOUT_NNODES` is a one-knob override to 5 → 1+5+2 if `starved_frac` ever spikes rollout-bound; default
  stays her validated 1+4+3.)
- **Seed dataset** auto-downloads at runtime from the public `chankhavu/ycchen-dsflash-proof-distill-v2-test`
  (a user-owned byte-faithful mirror). The node just needs HF network at pool init; to run fully offline,
  pre-build once: `python -m opd_v2.agentic.seed --run-dir <RUN_DIR>` → `<RUN_DIR>/pool/seed.jsonl`.
- **Teacher MoE backend is auto-selected** (`flashinfer_mxfp4` on B200) — no flag needed. Student rollout uses
  the `triton` attention-sink; trainer uses `olmo3_sink_fa2`. All validated on B200.
- **Checkpoints** land in `<RUN_DIR>/checkpoints/step_<N>/` every 25 steps (`CHECKPOINT_EVERY`, keep last 2):
  a DCP shard (exact resume) + a consolidated **bf16 HF** export in `step_N/hf/` (run
  `deploy/make_olmo3sink_deploy.py` on it before serving). Each write is a full DCP + ~64 GB HF export, so at
  25-step cadence budget ~2× the checkpoint I/O of a 50-step cadence.
- **Memory:** her knobs (`MEMFRAC 0.82`, `MICRO 131072`) were tuned at the ~140 GB (H200) edge; B200's 180 GB
  gives ~40 GB more headroom — nothing needs re-tuning.

Submit steps + launcher internals: [`README.md`](README.md) (this folder). Full pre-ship gate:
[`OPD_V2_SHIP_CHECKLIST.md`](../../../../docs/OPD_V2_SHIP_CHECKLIST.md).

---

## Appendix · Data flow & storage — what goes over HTTP vs shared disk

Four process types across the 8 nodes (rank 0 = teacher, 1–4 = rollout, 5–7 = trainer; rank 5 also runs the
CPU orchestrator). They exchange data over **two transports** — the rule of thumb is **token-ids / handles /
control go over HTTP; bulk tensors go over shared disk.**

### 1 · Over HTTP (control + small payloads only)
| from → to | call | payload |
|---|---|---|
| orchestrator → **rollout** | `POST /generate` | prompt token-ids → **generated token-ids** back (small) |
| orchestrator → **teacher** | `POST /score` | `{input_ids, out_path}` → a **handle** `{seq_len,…}` back. **The hidden-state bytes do NOT return over HTTP** — the teacher writes them to `out_path` on shared disk (see §2). |
| orchestrator → **trainer** | trainer HTTP (`trainer_endpoint.json`) | the training batch = token-ids + hidden **handles** (paths), + step control |
| orchestrator → **rollout** | `POST /update_weights_from_disk` | just a **trigger** — the rollout then reads the new weights from disk (§2) |
| head → all servers | `GET /health` | the health gate |

### 2 · Over shared disk = `RUN_DIR` (cross-node — must be ONE writable Weka mount at the SAME path on every node)
| channel | direction | path | size / churn |
|---|---|---|---|
| **hidden states** | **teacher → trainer** | `<RUN_DIR>/hidden/<uuid>.bin` | **LARGE + HIGH-CHURN** (~100–330 MB per trajectory, written → read once → deleted, continuously). This is the throughput-critical channel — it's the whole reason v2 moved hidden off HTTP. Put `RUN_DIR` on your **fast parallel FS**. |
| **weights (weight-sync)** | **trainer → rollout** | `<RUN_DIR>/weights/` (rolling `_a`/`_b`) | ~64 GB, rewritten every `WEIGHT_SYNC_EVERY` steps |
| durable checkpoints | trainer | `<RUN_DIR>/checkpoints/step_N/` | DCP shard + bf16 HF export, every 25 steps |
| coordination | all | `config.json`, `trainer_endpoint.json`, `.beaker_hosts_*`, agentic `pool/`, logs | small |

The trainer on rank 5 reads the `hidden/*.bin` the teacher wrote on rank 0 — so `RUN_DIR` **must** be writable,
shared, and the identical path on every replica. A read-only or non-shared mount breaks the loop at the first gather.

### 3 · Node-local (must NOT be on Weka)
| thing | env | why node-local |
|---|---|---|
| teacher hidden **staging** | `SGLANG_HIDDEN_SPOOL_DIR=/dev/shm/…` | intra-teacher-node only: the TP-worker ranks spool their hidden shards to `/dev/shm`, then the encode step reads them back **on the same node** and writes the consolidated file to the shared `<RUN_DIR>/hidden/`. This is why `sharedMemory: 128GiB` matters. Never point it at Weka. |
| JIT compile/autotune cache | `JIT_CACHE_DIR` | see the ⚠️ below. |

### 4 · `JIT_CACHE_DIR` concurrency (only relevant *if* a cache-related race is confirmed)
`JIT_CACHE_DIR` is the one path here that same-node same-role servers share (2 per role). The underlying JIT
libraries write their own caches atomically — and Yi-Chia's Slurm runs used the same shared dir with no issue,
so the compile/autotune sharing itself is fine. The **one** thing our cu128 image adds that she didn't have is
the **baked-seed `cp -rn`** into that shared dir (not atomic), so two same-node processes seeding at once is a
*possible* contention point — but this is a hypothesis, not a confirmed cause.

**If** a JIT-cache race is confirmed (check the actual error/path in `teacher_*`/`rollout_*.log`), the no-rebuild
sidestep is to **leave `JIT_CACHE_DIR` unset** — that skips our redirect+seed block entirely, reverting to stock
sglang's per-instance cache handling; you lose the fast warm-start (~15–25 min cold compile on the first launch)
but the run is unblocked. A one-line launcher change to make the seed-copy per-instance (so a shared path stays
race-safe *and* warm) is available on request. `RUN_DIR` — the real cross-node transport (§2) — is unaffected
either way.
