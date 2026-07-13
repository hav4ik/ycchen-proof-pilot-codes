# OPD v2 — pre-ship checklist (make the Ai2/Beaker version bulletproof)

Run this **top to bottom** before handing the image + Beaker launcher to Ai2. Every box must be checked on
the **exact pushed image** (not a local build, not a stale tag). The failure that motivated this doc is in
§0 — read it first.

## §0 — The lesson: image ↔ source DRIFT (this is how the bug happened)

A **built/pushed image lags the source** — commits merged *after* the build are NOT in the image, even
though they're on GitHub. Concretely:

- `chankhavu/ycchen-opd:cu128 @ sha256:5e3ba5f6…` was built at commit `74c1dac`.
- **`faf4c62`** (scale `AGENTIC_MAX_PROMPT_TOKENS` down for the smaller smoke `max_traj`) landed **after** →
  **not baked** → the scaled-smoke `agentic` run tripped the orchestrator guard
  `max_prompt_tokens=100000 > max_traj_tokens=57344`.
- Production (Beaker) was NOT affected (`env_v33` uses `max_traj=130816 > 100000`), but a shipped image must
  still contain every committed fix — you can't reason about "is it fixed?" from GitHub alone.

**Rule (non-negotiable): rebuild the ship image from the current `opd/b200-cu128` HEAD as the LAST step
before shipping, and prove no code/config commit is un-baked (§1).**

> ✅ **RESOLVED (2026-07-13):** the ship image was rebuilt from `94efb6c` → **`sha256:451201a8…`**, baking
> `faf4c62` (+ the seed mirror). Drift check `git log 94efb6c..HEAD -- docker/ training/ '*.sh' '*.py'` is
> **empty**, and both fixes were spot-checked *inside the container*. `451201a8` is the image that ships.

## §1 — Image is complete (no drift)

- [ ] **Rebuild from HEAD**, don't ship a stale tag:
      `DOCKER_BUILDKIT=1 docker build -f docker/cu128/Dockerfile.ycchen-opd -t chankhavu/ycchen-opd:cu128 .`
- [ ] **No un-baked code/config since the build** — this is the drift check that would have caught `faf4c62`:
      ```bash
      # <BUILD_COMMIT> = the commit you built from (record it!). Empty output = nothing un-baked.
      git log --oneline <BUILD_COMMIT>..HEAD -- docker/ training/ '*.sh' '*.py' Dockerfile*
      ```
- [ ] **Pushed digest recorded** and matches the docs (`OPD_V2_H200_SMOKE.md`, `OPD_V2_H200_BRINGUP_FIXES.md`).
- [ ] **Spot-check recent fixes are actually in the container** (not just the repo), e.g.:
      ```bash
      docker run --rm chankhavu/ycchen-opd:cu128 bash -lc '
        grep -c "AGENTIC_MAX_PROMPT_TOKENS" /opt/opd/launch/env_1node_smoke.sh   # faf4c62 -> 1
        grep -c JIT_CACHE_DIR /opt/opd/opd_serve/run_teacher.sh                  # -> 5
        grep -c "if not weight.is_cuda\|weight.dim() >= 2\|flash_rl_initial_load_complete\", False" \
          $(/opt/venv/serve/bin/python -c "import sglang,os;print(os.path.dirname(sglang.__file__))")/srt/model_loader/loader.py'  # -> 4
      ```

## §2 — Functional acceptance (all features, on the PUSHED image)

- [ ] Full finalization suite green — the 6-step checklist in
      [OPD_V2_H200_SMOKE.md#finalization-checklist](OPD_V2_H200_SMOKE.md#finalization-checklist-fresh-instance-baked-image):
      image integrity · sink (`test_attention_sink.py`) · **single_round** loop + checkpoint · resume ·
      **agentic** loop (dsflash seed) · JIT_CACHE_DIR persistence.
- [ ] **Both producers** run: `single_round` AND `agentic` (agentic is the production path — don't skip it).
- [ ] Weight-sync (3 flash_rl loader fixes), durable checkpoint + HF export, DCP resume, W&B online — all
      exercised, all green.
- [ ] Rollout sink parity (`serve_triton_parity --with-server`) PASS, or the loop's coherent trajectories as proxy.

## §3 — Beaker launcher + spec

- [ ] `beaker/opd_v33_b200.yaml` image ref points at the **current pushed digest**.
- [ ] **Every `<PLACEHOLDER>` filled**: cluster, budget, priority, image, NCCL `IB_HCA`/`SOCKET_IFNAME`,
      `sharedMemory`, and the Weka mounts — including a **WRITABLE** shared run dir (the loop coordinates
      through `RUN_DIR`; a read-only mount fails at the first gather).
- [ ] **3-node Beaker smoke passes** (`replicas: 3`, `TEACHER_NNODES=1 ROLLOUT_NNODES=1`, `MAX_STEPS=20`)
      BEFORE the full 64× B200 — it exercises the new launcher code (rank→role, hostname gather, cross-node
      c10d rendezvous, health gate) at 1/3 cost. See `beaker/README.md`.
- [ ] `BEAKER_NODE_HOSTNAME` peer-routability confirmed on the target cluster (the launcher's one unverified
      assumption — the health gate fails fast if wrong).
- [ ] **B200 teacher MoE backend = `flashinfer_mxfp4`** (VALIDATED 2026-07-13, baked in `env_v33_b200.sh` `88dce35`).
      DeepSeek-V4-Flash experts are fp4 → `auto`/`deep_gemm`/`flashinfer_trtllm` (all fp8 expert runners) CRASH on
      sm_100; `flashinfer_mxfp4` is the fp4-native path (same precision as her Hopper marlin). One-time ~15min fp4
      autotune on first launch (persisted). Rollout stays triton; trainer FA2. See `OPD_V2_H200_BRINGUP_FIXES.md`.
- [ ] **Seed source reachable at startup.** `SEED_SOURCE` now defaults to the user-owned public mirror
      `chankhavu/ycchen-dsflash-proof-distill-v2-test` (byte-faithful copy of `ycchen/…`, deletion-proof).
      `build_seed` does a live `load_dataset` at pool init, so the node needs HF network (public → no
      `HF_TOKEN`). For an offline cluster, pre-build the pool once and mount it:
      `python -m opd_v2.agentic.seed --run-dir <RUN_DIR>` → `<RUN_DIR>/pool/seed.jsonl` (skipped if present).

## §4 — Faithfulness (production == Yi-Chia's V33, exactly)

- [ ] `env_v33_b200.sh` matches her `run_agentic_mn_32b.sbatch` V33 — only the known, documented deltas
      (cu128 packaging, `olmo3_sink_fa2`, `triton` rollout, B200 MoE backend).
- [ ] **NO smoke-only knob leaks into production.** The smoke scales things DOWN; production must use HER
      values. Audit that `env_v33_b200.sh` does NOT set the scaled-down smoke values, and that production
      relies on the config.py defaults / her sbatch values:
      | knob | smoke (env_1node_smoke.sh) | production (her V33) |
      |---|---|---|
      | `AGENTIC_MAX_PROMPT_TOKENS` | `MAX_TRAJ/2` | **100000** (config default; env_v33 must NOT set it) |
      | `MEMFRAC` | 0.70 (140GB H200) | **0.82** (180GB B200) |
      | `WEIGHT_SYNC_EVERY` | 1 | **4** |
      | `TRAIN_BATCH_TRAJS` | 4 | **64** |
      | `MAX_TRAJ_TOKENS`/`MICRO` | 57344 | **130816 / 131072** |
      | bundle caps | 8k / 8k | **40k / 50k** |
      | `CHECKPOINT_EVERY` | 0 (smoke) / 5 (test) | **50** |
- [ ] Her training config intact: `LR=1e-5`, `constant` schedule (no warmup), `β=1.0`, adam `(0.9, 0.95)`,
      `MAX_STEPS=100000`, `max_staleness=0`.
- [ ] **Teacher serve backend = hers, unmodified.** `run_teacher.sh` keeps `MOE_BACKEND=auto` (the stock
      `apply_deepseek_v4_defaults` hook picks `dsv4` attention + fp8-e4m3 KV + a DeepGEMM fp8 MoE on sm100)
      and does **NOT** enable `--enable-deepseek-v4-fp4-indexer`. Rationale (locked 2026-07-13): the teacher
      runs **prefill-only** (`/score` = one forward pass, no `max_new_tokens`/decode), so the fp4-indexer
      cache's benefit (compressing a *growing decode* cache) doesn't apply — it would only add a sparse-top-k
      precision perturbation to the distillation target. A teammate's vLLM `deep_gemm_mega_moe` +
      `use_fp4_indexer_cache` is a decode-oriented throughput lever, not a fit for our prefill-only teacher.
      Both remain **benchmark-only overrides** (`MOE_BACKEND=deep_gemm`; fp4 indexer not wired) — adopt only
      on a real throughput win **and** byte-exact `/score` parity, and flag as a deviation.

## §5 — Ops (so the B200 run doesn't stall on cold start)

- [ ] `JIT_CACHE_DIR` → a **persistent Weka path** (fixed, not per-run `RUN_DIR`) so DeepGEMM/flashinfer/etc.
      compile caches survive across runs and instances. Arch-scoped (`sm100` for B200).
- [ ] DeepGEMM pre-warm plan: either the persisted `JIT_CACHE_DIR`, or `python3 -m sglang.compile_deep_gemm`
      AOT before the first run — so the teacher doesn't eat the ~10-20 min cold compile on the first B200 node.

## §6 — Known image↔source deltas (keep current)

| image | built at | functional commits missing | mitigation |
|---|---|---|---|
| **`cu128 @ 451201a8`** (SHIP) | **`94efb6c`** | **none** — drift-clean | none needed; this is the image for Ai2 |
| `cu128 @ 5e3ba5f6` (superseded) | `74c1dac` | `faf4c62` (agentic auto-scale) | replaced by `451201a8`; do NOT ship |
| `cu128-flashinfer-sink @ 30994b4d` | `321aff6` | `faf4c62` + seed mirror | not shipped to Ai2 (triton is production); rebuild that branch if ever needed |

> **Ship rule of thumb:** the number that goes to Ai2 is the digest of an image built from a commit where
> `git log <that-commit>..HEAD -- docker/ training/ *.sh *.py` is **empty**. If it's not empty, rebuild.

> ⏳ **PENDING FINAL REBUILD (B200 bring-up):** `451201a8` (built @`94efb6c`) is missing `8240b78`
> (`test_attention_sink.py` venv-order fix — **test-only**, does not affect training; production `run_*`
> scripts already invoke the trainer python directly, and Ai2's cuda-12.8 B200 selects the venv correctly
> anyway). B200 component testing may surface more small fixes — **batch them into one final drift-clean
> rebuild before the Ai2 handoff**, then re-stamp the digest here. `451201a8` remains valid for B200 testing
> (use `python /opt/opd/opd_v2_train_smoke.py` directly instead of the `-k fa2` harness on native-cuda-13 nodes).
