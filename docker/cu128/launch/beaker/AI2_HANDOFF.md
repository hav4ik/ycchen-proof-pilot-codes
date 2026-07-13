# OPD v2 — Ai2 operator handoff (what YOU need to provide)

Everything else — the training/serving code, both venvs, Yi-Chia's patches, the launchers, the teacher's
`flashinfer_mxfp4` MoE auto-detect — is **baked into the image**. To run the loop on your Beaker cluster you
provide exactly three things: **(1) the 2 model checkpoints in the right mount paths, (2) a WRITABLE shared
run dir, and (3) a WRITABLE shared JIT-compile cache dir.** Then fill the site-specific placeholders and submit.

**Image (ship digest, drift-clean, all B200 fixes validated on B200):**
```
chankhavu/ycchen-opd:cu128@sha256:4c0d4276dc45fe21877dc0d6d028887f4c8a6d91ae91e102eb90c262b3726222
```
(Optionally import it into Beaker for faster pulls than Docker Hub, then use the `beaker:` image field.)

**In this folder:** [`README.md`](README.md) (submit steps + launcher internals) · [`opd_v33_b200.yaml`](opd_v33_b200.yaml)
(production spec, 8×B200, her V33 1+4+3) · [`opd_smoke3_b200.yaml`](opd_smoke3_b200.yaml) (3-node launcher smoke).
**This file is the prerequisites + spec-filling guide.**

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
- [ ] *(optional)* Pre-warm `JIT_CACHE_DIR` from [`chankhavu/opd-jit-cache-sm100`](https://huggingface.co/datasets/chankhavu/opd-jit-cache-sm100) to skip the ~10–20 min first-launch compile — see §2.
- [ ] Seed: nothing (auto-fetched at runtime) — OR pre-build `<RUN_DIR>/pool/seed.jsonl` for an offline cluster.
- [ ] The 3 team-specific yaml values filled (budget, Weka bucket+subPath, priority) — see §3.
- [ ] `WANDB_API_KEY` set as a Beaker secret (optional; W&B is online by default).

## 2 · Provide the three writable/shared paths

| path (in-job) | must be | why |
|---|---|---|
| `/models/DeepSeek-V4-Flash`, `/models/student-deploy` | mounted (read-only OK) | the checkpoints from step 1 |
| **`RUN_DIR`** (e.g. `/weka/run/opd_v33`) | **WRITABLE + shared + identical path on every replica; a DISTINCT path per run (smoke ≠ production)** | the loop's single source of truth: `config.json`, trainer endpoint, teacher hidden-state spool index, weight-sync buffer, rolling weights, **DCP + HF checkpoints**, the rank→hostname gather, all logs. A read-only mount fails at the first gather. The smoke and full run **must not share** it (the smoke's scaled-down `config.json` + short-context pool would clobber production's) — the two yamls already default to different paths (`/weka/run/opd_smoke3_b200` vs `/weka/run/opd_v33_b200`). |
| **`JIT_CACHE_DIR`** (e.g. `/weka/jit_cache`) | **WRITABLE + shared + FIXED (not per-run)** | the DeepGEMM/triton/flashinfer JIT-compile cache. The serve stack **writes** compiled kernels here. Make it a **fixed** path (NOT under `RUN_DIR`) so it **persists across runs and instances**. Optionally pre-warm it from [`chankhavu/opd-jit-cache-sm100`](https://huggingface.co/datasets/chankhavu/opd-jit-cache-sm100) (see below). |

**About the JIT cache (your "writable/saveable cache dir"):** the DeepSeek-V4 teacher JIT-compiles fp8/fp4
kernels **per GEMM shape** — a cold node spends **~10–20 min** on DeepGEMM + **~15 min** on the flashinfer fp4
autotune at first launch. With a writable, persistent `JIT_CACHE_DIR` this is a **one-time** cost: the run
scripts arch+role-scope it (`sm100/{teacher,rollout}/…`), so the first replica compiles and **every later
replica + future run reuses it**. Point it at Weka so it survives job restarts.

### Optional: pre-warm the JIT cache from the public dataset (skip the first cold compile)
We publish a pre-built sm_100 / TP4 JIT **compile** cache — **[`chankhavu/opd-jit-cache-sm100`](https://huggingface.co/datasets/chankhavu/opd-jit-cache-sm100)**
(one 20 MB artifact, `opd-jit-sm100-tp4.tgz`). Extract it into your `JIT_CACHE_DIR` **once, before the first
run**, and the very first launch skips the ~10–20 min DeepGEMM/JIT compile (the run scripts find a warm cache):
```bash
# on a networked prep node, with JIT_CACHE_DIR set to your fixed Weka cache path:
hf download chankhavu/opd-jit-cache-sm100 opd-jit-sm100-tp4.tgz --repo-type dataset --local-dir /tmp/jitseed
mkdir -p "$JIT_CACHE_DIR"
tar xzf /tmp/jitseed/opd-jit-sm100-tp4.tgz -C "$JIT_CACHE_DIR"   # -> $JIT_CACHE_DIR/sm100/{teacher,rollout}/…
```
Notes: (1) it's the **compile** cache (deep_gemm/triton/flashinfer/sglang/tvm-ffi/inductor cubins), which is
portable; the flashinfer **fp4 autotune** is env-specific so it's *not* included — the teacher still runs its
~15 min autotune on first launch, after which it too persists in your writable `JIT_CACHE_DIR`. (2) It is
**sm_100 (B200/B300) at TP4** — matches this loop's `TEACHER_TP=ROLLOUT_TP=4`; a different TP or GPU arch just
recompiles from scratch (harmless). (3) Purely optional — it only saves cold-start minutes; the writable
`JIT_CACHE_DIR` is what actually matters. Opt out of any baked seed with `JIT_CACHE_SEED=0`.

## 3 · Fill the Beaker spec — most of it is already done

Both `docker/cu128/launch/beaker/opd_v33_b200.yaml` (production) and `opd_smoke3_b200.yaml` (smoke) are
**pre-filled against the Beaker docs** — cluster, NCCL, GPU count, shared memory, timeout, and the pinned image
are already set for **Titan**:

- **Cluster: `ai2/titan-cirrascale`** — 96× **B200 (192 GB)**, 8× IB @ 400 Gbps/GPU. Titan requires
  **PyTorch ≥2.7 + CUDA 12.8+**; our image satisfies it (trainer torch 2.10+cu128, serve torch 2.11+cu130).
  Her 8-node V33 uses **64 of the 96 GPUs**; the 3-node smoke uses 24. (Alt: `ai2/holmes` = 576× B300 (288 GB) —
  also sm_100, the teacher auto-detect handles it; uncomment the line.)
- **NCCL** pre-set to the Ai2 IB values: `NCCL_SOCKET_IFNAME=ib`, `NCCL_IB_HCA=^=mlx5_bond_0`, `NCCL_DEBUG=INFO`.
- **`gpuCount: 8`**, **`sharedMemory: 128GiB`**, **`timeout`** — set. Image pinned to the ship digest.
- **Weka is read-write at Ai2** (jobs write as `root:root`), so `RUN_DIR` + `JIT_CACHE_DIR` under a Weka mount
  are writable — no special config needed.

**You only fill THREE team-specific values** (each marked `<PLACEHOLDER: …>`, present in both yamls):
1. **`budget`** — your team's budget account, e.g. `ai2/oe-training`.
2. **Weka `weka: <bucket>` + `subPath:`** — your team's bucket (see https://weka.allen.ai/), e.g.
   `oe-training-default`, for the three mounts: the writable run dir (`RUN_DIR` + `JIT_CACHE_DIR` live under it)
   and the two model dirs from step 1. Also set the `RUN_DIR` / `JIT_CACHE_DIR` env values to paths **under**
   that mount (defaults `/weka/run/opd_v33` and `/weka/run/jit_cache` assume a mount at `/weka/run`).
3. **`context.priority`** — your allocation tier on Titan (strict-priority cluster): `low|normal|high|urgent`.

W&B is online by default — set `WANDB_API_KEY` from a Beaker **secret** (`beaker secret write wandb-api-key <key>`,
then uncomment the `secret:` line). `HF_TOKEN` is **not** needed (models + seed dataset are public).

## 4 · Submit: smoke first, then the full run

```bash
# (a) 3-node launcher smoke — validates rank->role, the shared-FS hostname gather, cross-node c10d, health gate
beaker experiment create docker/cu128/launch/beaker/opd_smoke3_b200.yaml
#     green = all 3 replicas gather hostnames, teacher+rollout health-pass, trainer forms world 8,
#             steps 1..20 with loss ↓, weight-sync ticking (footer of the yaml has the full checklist)

# (b) full 64x B200 V33 — her production config
beaker experiment create docker/cu128/launch/beaker/opd_v33_b200.yaml
```
The smoke (`RUN_DIR=/weka/run/opd_smoke3_b200`) and the full run (`RUN_DIR=/weka/run/opd_v33_b200`) write to
**separate** run dirs — keep them distinct so the smoke's scaled-down config/pool never touches production.
They **do** share one dir on purpose: `JIT_CACHE_DIR` (`/weka/run/jit_cache`), so the smoke warms the fp4/DeepGEMM
kernels that the full run then reuses.

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
