# OPD v2 — H200 smoke bring-up: bugs hit & fixes

Chronological log of the real bugs found bringing the cu128 image up on a live **8×H200 (driver 570.195,
CUDA 12.8)** instance, and how each was fixed. Each failure was *further down the pipeline* than the last
— useful both as a record and as a triage guide if the next node behaves differently.

## Environment that surfaced these

- **Node:** rented 8×H200, NVIDIA driver **570.195.03 → max CUDA 12.8** (the same ceiling as the Ai2/Beaker
  target). GPUs run *inside* the image container (no host `docker run` wrapper).
- **Image:** `chankhavu/ycchen-opd:cu128` — base/**trainer** venv is genuine cu128 (torch 2.10); the
  **serve** venv (sglang) is cu130 (see fix #1). Current SHIP digest (fixes #1–#10 + JIT_CACHE_DIR + `faf4c62`
  agentic auto-scale + seed mirror; built at `94efb6c`, drift-clean):
  `sha256:451201a83fb4fdeae25a52fc387c3420000dc84fbd27565c12a1b145629bd5bd`.

## Fix #1 — serve stack is CUDA-13; driver 570 is CUDA-12.8 → forward-compat

- **Symptom:** both teacher and rollout die in ~10 s during CUDA init:
  `The NVIDIA driver on your system is too old (found version 12080)` → `RuntimeError: No accelerator … available`.
- **Root cause:** sglang **0.5.14** is inherently a **CUDA-13** build — it hard-pins `cuda-python>=13.0`
  and `torch==2.11.0` (which resolves to `+cu130`). This is **not** our drift: **Yi-Chia's own
  `0.5.12.post1` is identical** (`torch==2.11.0`, `cuda-python>=13.0`); the whole `0.5.11–0.5.14` band is
  CUDA-13. It **cannot** be pinned to cu128 — `torch cu128` needs `cuda-bindings<13` while sglang's
  `cuda-python` needs `~=13.x` → `ResolutionImpossible` (community-confirmed: **sgl-project/sglang#25069**;
  the maintainer's cu128 workaround silently downgrades below 0.5.11). She ran her CUDA-13 sglang on an
  H200 whose driver was CUDA-13-capable; our only difference is the 12.8 driver ceiling.
- **Fix (`fd934a5`):** keep sglang 0.5.14 untouched (the attention-sink patches are re-anchored to it) and
  bake NVIDIA's **official CUDA-13 forward-compat `libcuda`** (multi-stage COPY from
  `nvidia/cuda:13.0.1-base-ubuntu22.04` → `/opt/cuda13-compat`). `run_{teacher,rollout}.sh` load it onto
  `LD_LIBRARY_PATH` **only when the node driver is < CUDA 13** (datacenter GPU, driver ≥ 525). On a ≥580
  (CUDA-13) driver — e.g. B200 hosts — it stays inert and the native driver is used. The trainer venv is
  genuine cu128 and runs natively on the 570 driver.
- **Status:** ✅ **confirmed on real H200 silicon** — `rollout.log` + `teacher.log` both print
  `[cuda13-compat] node driver CUDA 12.x < 13 -> forward-compat libcuda enabled` and get past CUDA init.

## Fix #2 — `rope_theta` KeyError loading the student (olmo2.py)

- **Symptom:** with the driver fixed, the rollout now dies ~30 s in, during model construction:
  `File …/sglang/srt/models/olmo2.py, line 116 … config.rope_parameters["rope_theta"]` → `KeyError: 'rope_theta'`.
- **Root cause:** the deploy student config carries `rope_theta: 500000` (top-level) + `rope_scaling: {yarn}`
  — nothing is missing. But sglang 0.5.14 ships a **custom `configs/olmo3.py`** that exposes
  `config.rope_theta = 500000` and fills `config.rope_parameters` with **only the yarn scaling params
  (no `rope_theta`)**. Stock `olmo2.py` reads `rope_theta` from `rope_parameters` → KeyError. The two
  sglang files are internally inconsistent for a legacy-rope config. **Not a version issue** — 0.5.12.post1
  and 0.5.14 share the same `olmo2.py` line, so downgrading would not have helped.
- **Fix (`0b90d30`, corrected in `aca02c8`):** in our `olmo2.py` overlay, read
  `rope_parameters["rope_theta"]` if a config actually nests it, else fall back to the real
  `config.rope_theta` attribute — **no invented default** (an early `10000.0` fallback was removed: if the
  value were ever truly absent it must raise loudly, not silently mis-scale). Verified against sglang's
  `get_config()`: old path KeyErrors, new path yields **500000**, `rope_scaling` (yarn) untouched for `get_rope`.
- **Status:** ✅ fixed & value-verified; baked in the current digest.

## Fix #3 — teacher `/health` gate times out on cold DeepGEMM JIT

- **Symptom:** teacher never passes the health gate; the server self-warmup POST times out at 600 s while
  fp8 kernels JIT-compile on a cold cache.
- **Root cause:** sglang self-warmup fires a warmup request with a fixed timeout; DeepSeek-V4 fp8 GEMM
  kernels JIT-compile per-shape on first use (cold ~100 s+ across shapes) → exceeds it on a cold node.
- **Fix:** `--skip-server-warmup` on both teacher scripts — the loop warms the teacher via real `/score`
  traffic anyway. DeepGEMM cache persists to `~/.cache`, so later starts are fast.
- **Status:** ✅ teacher reaches healthy; `/score` 200s confirmed byte-exact.

## Fix #4 — serve JIT needs a CUDA-13 *toolkit*, not just the CUDA-13 driver

- **Symptom:** teacher crashes JIT-compiling the DeepSeek indexer top-k / DeepGEMM kernels (nvcc /
  `cuda::ptx` intrinsic errors); the base image's cu128 nvcc can't build them.
- **Root cause:** the CUDA-13 serve stack JIT-compiles kernels with CUDA-13 intrinsics/headers. Fix #1's
  forward-compat `libcuda` fixes the *driver*; the *compiler* toolchain must also be CUDA-13. (`nvcc` ≠
  driver.) Note: this is the **DSA indexer top-k**, not MoE routing — it fires regardless of the teacher's
  "no Top-K" MoE claim.
- **Fix:** bake `cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0`; set `CUDA_HOME=/usr/local/cuda-13.0`
  in the serve run scripts (trainer venv stays cu128).
- **Status:** ✅ top-k / DeepGEMM compile.

## Fix #5 — flashinfer sampling JIT: `curand.h: No such file or directory`

- **Symptom:** rollout (and any sampler) dies JIT-compiling flashinfer's `sampling`/`renorm` kernel:
  `fatal error: curand.h: No such file or directory`.
- **Root cause:** flashinfer's sampling kernel `#include <curand.h>`, but the minimal CUDA-13 toolkit
  (nvcc+cudart+cccl, fix #4) ships **no math-library dev headers**.
- **Fix (`d03e7eb`):** add `cuda-libraries-dev-13-0` (cuRAND/cuBLAS/cuSOLVER/cuSPARSE dev) — covers curand
  plus any other flashinfer/sglang JIT math dep in one shot; build asserts `curand.h` present.
- **Status:** ✅ (shared serve venv → fixes both rollout and teacher).

## Fix #6 — `max_prefill_buffer_tokens` AttributeError (server_args overlay drift)

- **Symptom:** serve crash: `AttributeError: 'ServerArgs' object has no attribute 'max_prefill_buffer_tokens'`.
- **Root cause:** our `server_args.py` overlay (re-anchored to 0.5.14) dropped the stock
  `max_prefill_buffer_tokens` method. **First overlay re-derivation to silently drop a stock member** —
  see fix #10 for the second (and worse) instance of this class of bug.
- **Fix:** re-added the stock 0.5.14 method (with a local `import math`) to the overlay.
- **Status:** ✅.

## Fix #7 — `single_round` prompt template not found

- **Symptom:** orchestrator dies: `FileNotFoundError … proofbench_generator.txt`.
- **Root cause:** the vendored prompt loader (`_vendor_opd/opd/prompts.py`) resolves the template dir one
  segment too high → `/opt/opd/distill_gen` instead of `/opt/opd/repo/distill_gen`.
- **Fix (`1634689`):** Dockerfile symlink `/opt/opd/distill_gen → /opt/opd/repo/distill_gen`.
- **Status:** ✅ — loop reached `step=1` after this.

## Fix #8 — rollout "0.25 tok/s" was teacher *starvation*, not a forward-compat tax (misdiagnosis corrected)

- **Symptom:** rollout throughput logged ~0.25 tok/s; I successively (mis)blamed OOM, then cuda-graph, then
  the forward-compat `libcuda`.
- **Root cause:** the teacher was crashing (fixes #3–#6), so the rollout sat **starved** waiting on `/score`,
  not running slow. A standalone rollout benchmark is healthy: **91.7 tok/s** (cuda-graph, batch-1) and
  **279 tok/s** aggregate at batch-4 in the live loop. The forward-compat layer imposes **no large tax on
  the cuda-graph path** (its overhead is at the kernel-launch boundary, hidden by graph replay).
- **Status:** ✅ rollout confirmed healthy; the "~3× forward-compat tax" framing is **retracted** — batch-1
  / TP2 memory+comm limits explain the numbers.

## Fix #9 — weight-sync OOMs the rollout (`MEMFRAC` too high for the H200 fp8-reload peak)

- **Symptom:** at the first weight sync, the rollout OOMs in `update_weights_from_disk` (`.clone()`), plus a
  secondary `per_token_group_quant_8bit` CPU `NotImplementedError` in the crash teardown.
- **Root cause:** the fp8 reload peak is **~18–26 GB** (old fp8 copy + loader clone + bf16 re-quant — her
  `run_agentic_mn_32b.sbatch:73`). `MEMFRAC=0.85` on a **140 GB H200** left only ~5 GB free (fp8 weights sit
  *on top of* the static KV pool → process at 135/140 GB). Her `0.82` is sized for a **180 GB B200**.
- **Fix (`8313c72`):** smoke default `MEMFRAC=0.70` (~25 GB free, matching her B200 headroom scaled to H200)
  + `WEIGHT_SYNC_EVERY=1` so a plumbing run exercises the transfer at step 1, not step 4.
- **Status:** ✅ OOM cleared (`avail mem=37.74 GB`) — which then exposed fix #10.

## Fix #10 — the baked `loader.py` re-derivation dropped **3** of her flash_rl fixes

- **Symptom:** with the OOM gone, the reload *still* crashes:
  `NotImplementedError: 'sglang::per_token_group_quant_8bit' … 'CPU' backend`, and `load_weights_proxy`
  appears **twice** in the traceback (the reload is nesting). **This was NOT an OOM symptom** — my earlier
  claim that it was is retracted; it reproduces with 37 GB free.
- **Root cause:** the Dockerfile (`:166`) bakes **`docker/cu128/opd_serve/sglang_patches/model_loader/loader.py`**
  — our 0.5.14 re-derivation — into the serve venv, **not** her `training/opd_v2/flash_rl/patches/loader.py`.
  The re-derivation silently dropped **all three** of her proof-pilot flash_rl fixes:
  1. **CPU→CUDA guard** — `update_weights_from_disk` yields CPU tensors, but `per_token_group_quant_fp8` is a
     CUDA-only fused kernel → move to the current device before quantizing. *(the crash you see)*
  2. **`dim >= 2` guard** — 1-D Olmo3 norms (`q_norm`/`k_norm`/`post_feedforward_layernorm`) crash the 2-D
     quant kernel → restrict quant to ≥2-D so 1-D params fall through to keep. *(the next crash)*
  3. **nested-proxy reload guard** — `load_weights_and_postprocess` is re-entered by every sync; without an
     early-return it re-wraps the already-installed `load_weights` proxy, so the Nth reload recurses N levels,
     each re-materialising `list(weights)` + re-quantizing every weight → **linear memory blowup** (also
     contributed to fix #9's OOM; the double proxy in the stack is its fingerprint).
- **Fix (`d25acfa`, `6e2a485`):** ported all three **verbatim** from her source. Audit is now clean — both
  her proof-pilot sites are present in the overlay, and `SKIP_QUANTIZATION_PARAMS` + `is_reload_scenario`
  are byte-identical to hers.
- **Lesson (important):** the two `loader.py` files are for **different sglang versions**; the docker overlay
  is a re-derivation and MUST be audited against her source for **every** proof-pilot marker. This is the
  **third** overlay to drop fixes (server_args #6, and both fixes #1+#2 of the quant path + the proxy guard).
  Any future overlay re-anchor needs the same `grep -n "proof-pilot"` parity check.
- **Status:** ✅ committed + hot-patched in-container; **pending image rebuild** to bake.

## Also fixed during the B200 launcher audit (not a smoke crash)

- **Teacher dist-init faithfulness (`659171f`):** `run_mn_cu128.sh` set `DIST_INIT_ADDR=127.0.0.1:$dist`
  (forcing loopback), whereas her `run_mn.sh` sets `DIST_INIT_PORT` (which the teacher script doesn't read
  → sglang auto-picks). Her delivered V33 ran with auto-pick, so ours was an unverified divergence, not a
  bug fix → reverted to `DIST_INIT_PORT` to match her exactly. Multi-node path only; unrelated to the H200
  smoke.

## Operational gotchas (not code bugs — worth knowing)

- **Run *inside* the container.** On these instances you're already in the image (`root@…:/opt/opd/repo…`),
  so skip the docs' `docker run …` wrapper and run `source env_1node_smoke.sh && bash run_1node.sh` directly.
- **`--local-dir` is literal.** `hf download … --local-dir /models/…` ignores `$MODELS`; set `MODELS=/models`
  and use `/models/...` consistently (a `/data/models` vs `/models` mix-up made an `ls` "fail" while the
  models were fine).
- **Pull the *new* digest.** A cached image without `/opt/cuda13-compat` reproduces fix-#1's crash — verify
  `ls /opt/cuda13-compat/libcuda.so*` after pulling, or pull by digest.

- **⚠️ Validate the serve with the CHAT endpoint, not raw `/generate` — this cost us a full afternoon.** The deploy
  student is a **reasoning/chat model**. Hitting `POST /generate` with `{"text": "..."}` (raw completion, no chat
  template) is **out-of-distribution**, and the model degenerates — repetition, single-token collapse, mid-sentence
  language-switching (e.g. `二十一th`). **This is NOT a serving or hardware bug.** Test via
  `POST /v1/chat/completions` with `messages` + `temperature: 0` (applies the chat template + the
  `--reasoning-parser deepseek-r1` scaffold the server sets) → a clean IMO-level Euclid proof
  (`reasoning_content` + a `\boxed{}` answer, `finish_reason: stop`). The **real loop is unaffected** — the
  producer builds `input_ids` *with* the prover/chat template, which is why the H200 loop trained fine (eos=100%).
  **Cautionary tale:** the raw-`/generate` garbage *appeared* to correlate with GPU arch and CUDA driver (sm_100
  vs sm_90; native cuda-13 vs cuda-12.8 forward-compat), and we nearly committed a "native cuda-13 serve is broken,
  require a driver-570 node" theory + burned three rentals on it. It was **sampling noise on OOD input** — on the
  **H200 whose raw `/generate` had collapsed (driver 595, native cuda-13.2)**, the chat endpoint returns a correct
  Euclid proof. So serve correctness is a **prompt-format** issue, not a driver/arch one (confirmed on **B200 sm_100**
  too: clean chat-endpoint proof). Always reproduce a suspected serve bug through the chat endpoint before blaming
  the stack.

- **B200 (sm_100) student side — VALIDATED (2026-07-13).** *Trainer FA2 sink:* `python /opt/opd/opd_v2_train_smoke.py`
  → PASS (fp64-exact sink correction, **bit-exact** OPD JSD loss+grad, forward/sink-grad/q-k-v-grad parity within
  bf16 tol, `torch.compile(fullgraph=True)` clean, packed doc-isolation 0 leak) — the FA2 wheel's **sm_100** kernel
  is correct. *Rollout:* fp8 weights (flash_rl) + fp8-KV + triton sink → clean IMO-level Euclid proof via the chat
  endpoint on B200 (`finish_reason: stop`). Run the trainer smoke **directly** (bare `python` = the `/opt/conda`
  **cu128 trainer venv**); the `test_attention_sink.py -k fa2` wrapper mis-picks the cu130 *serve* venv on
  native-cuda-13 nodes (its `cuda==12.8` guard then trips) — harness fix in `8240b78`, pending next rebuild.
  *Remaining B200 component: the DeepSeek-V4-Flash teacher (TP4).*
- **`Scale param shape … not divisible by 3` during weight-sync is BENIGN.** It's her loader (identical at
  `flash_rl/patches/loader.py:1087` / overlay `:1141`): the fused qkv scale dim isn't a clean 3× multiple
  because GQA makes q/k/v different sizes. The `rows_per_shard = dim//3` estimate that triggers the warning
  is only used to *skip missing* shards; present shards are placed at their correct offset by their **actual**
  size (`shard_scale.shape[0]`), so nothing is mis-scaled. Cosmetic — fires in her prod too; left as-is.

- **"Teacher prefill is slow / inconsistent" is usually a metric artifact + a config choice, not a teacher bug.**
  Two things conspire on a cold node:
  1. **The `input throughput (token/s)` on the FIRST chunk of each `/score` is idle-polluted** — sglang computes
     it as `tokens ÷ time-since-last-activity`, which includes the wait for the rollout to deliver the next
     trajectory. Low numbers there (30–200 tok/s) do **not** mean slow prefill: the same-second remainder chunk
     shows the real rate (10–40k tok/s). Sanity check: `first-chunk-tokens ÷ idle-gap ≈ the shown number` (e.g.
     `11264 ÷ 54 s = 208` matched a logged `206.28`). If the low chunk were real it'd take minutes, but the next
     chunk logs 0 s later — so it didn't.
  2. **Small prefills are launch-bound.** The teacher is DeepSeek-V4 (MoE+MLA+DSA) run **eager**
     (`--disable-cuda-graph`, hers — correct: cuda-graph is a decode opt, prefill has varying shapes), so a
     forward pass is **thousands of small kernel launches**. On the **forward-compat** driver each launch pays an
     extra per-launch cost; for **small** prefills (short trajectories → small `MAX_NEW_TOKENS`) that cost isn't
     hidden behind compute → the GPU is launch-starved. Also DeepGEMM JIT-compiles **per GEMM shape** (M = prefill
     length), and MoE routing varies per-expert M, so short/varied trajectories keep hitting new shapes → ~8 min
     compile stalls (heartbeat freezes). **LONG/uniform trajectories** chunk to a constant **11264** compute-bound
     prefill → launch overhead amortized + one dominant shape → 12–40k tok/s (verified). The rollout is unaffected
     because it uses cuda-graph (per-launch cost paid once at capture).
  **Fixes, cheapest first:** (a) long trajectories (compute-bound); (b) **`JIT_CACHE_DIR`** on a persistent mount
  → compile cache survives runs+instances (baked, opt-in — see run_{teacher,rollout}.sh); (c)
  `python3 -m sglang.compile_deep_gemm` AOT pre-compile → zero runtime JIT even cold; (d) a native **≥580 driver**
  → removes the forward-compat launch tax entirely.

## Quick reference

| what | value |
|---|---|
| Her sglang | **0.5.12.post1** (CUDA-13: torch 2.11 + cuda-python≥13) |
| Our sglang | 0.5.14 (same CUDA-13 generation) |
| cu128 sglang cutoff | naive `pip` → cu130; but LMSys ships an official **`v0.5.14-cu129`** and cu128 is self-buildable (see [OPD_V2_CU129_REBASE](#) / memory) — a cleaner future base than cu130+forward-compat |
| Image = | cu128 **trainer** + cu130 **serve** (forward-compat on <13 drivers) |
| CUDA↔driver | 12.8→≥570 · 12.9→≥575 · 13.0→≥580 · forward-compat→≥525 |

## Failure ladder (triage aid)

```
CUDA init "driver too old"        → fix #1  (forward-compat libcuda)           [✅]
model build KeyError rope_theta   → fix #2  (olmo2.py rope read)               [✅]
teacher health-gate timeout       → fix #3  (--skip-server-warmup)             [✅]
serve JIT topk/DeepGEMM           → fix #4  (CUDA-13 nvcc/cccl toolkit)        [✅]
flashinfer JIT curand.h           → fix #5  (cuda-libraries-dev-13-0)          [✅]
server_args AttributeError        → fix #6  (max_prefill_buffer_tokens re-add) [✅]
prompt template FileNotFound      → fix #7  (distill_gen symlink)              [✅]
rollout "0.25 tok/s"              → fix #8  (teacher starvation, NOT fwd-compat)[✅]
>>> step=1 train/loss             → ✅ REACHED (loss 0.0948, gnorm 0.25, eos 100%, rKL live, g4 0.900)
weight-sync OOM (rollout)         → fix #9  (MEMFRAC 0.85→0.70 for H200 VRAM)  [✅]
weight-sync CPU NotImplementedErr → fix #10 (loader.py: 3 dropped flash_rl fixes ported) [✅ committed + hot-patched; pending rebuild]
step → 10, rc=0                   → (in progress)
Shot 2 (agentic + dsflash, 57k)   → (next)
```

**Milestone:** the full 4-process loop turned end-to-end on 8×H200 at `step=1` (rollout fp8 + FA2/triton
sink → teacher `/score` hidden-extract → reverse-KL → FSDP2 step). Fixes #9–#10 are the weight-sync edge
(trainer→rollout), the last untested part of the loop.

**Recurring failure class — overlay re-derivation drift:** three of these (server_args #6, and the
CPU→CUDA + dim≥2 + nested-proxy fixes in #10) are our 0.5.14 *overlays* silently dropping members/fixes
present in her source. Any overlay re-anchor MUST be diffed against her source; for `loader.py`,
`grep -n "proof-pilot"` must match between `docker/cu128/opd_serve/sglang_patches/model_loader/loader.py`
and `training/opd_v2/flash_rl/patches/loader.py`.
