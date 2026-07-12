# OPD v2 — H200 smoke bring-up: bugs hit & fixes

Chronological log of the real bugs found bringing the cu128 image up on a live **8×H200 (driver 570.195,
CUDA 12.8)** instance, and how each was fixed. Each failure was *further down the pipeline* than the last
— useful both as a record and as a triage guide if the next node behaves differently.

## Environment that surfaced these

- **Node:** rented 8×H200, NVIDIA driver **570.195.03 → max CUDA 12.8** (the same ceiling as the Ai2/Beaker
  target). GPUs run *inside* the image container (no host `docker run` wrapper).
- **Image:** `chankhavu/ycchen-opd:cu128` — base/**trainer** venv is genuine cu128 (torch 2.10); the
  **serve** venv (sglang) is cu130 (see fix #1). Current digest after these fixes:
  `sha256:a3d7c1802598d6699920d2f9910703be0e19d09dc4bd80335f4fc21a55dd8d73`.

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

## Quick reference

| what | value |
|---|---|
| Her sglang | **0.5.12.post1** (CUDA-13: torch 2.11 + cuda-python≥13) |
| Our sglang | 0.5.14 (same CUDA-13 generation) |
| cu128 sglang cutoff | **≤ 0.5.10** only (would require re-porting the sink patches — not done) |
| Image = | cu128 **trainer** + cu130 **serve** (forward-compat on <13 drivers) |
| CUDA↔driver | 12.8→≥570 · 12.9→≥575 · 13.0→≥580 · forward-compat→≥525 |

## Failure ladder (triage aid)

```
CUDA init "driver too old"  → fix #1 (forward-compat)          [✅ cleared on H200/570]
model build KeyError rope_theta → fix #2 (olmo2.py rope read)  [✅ cleared]
model weights load / GPU init   → (next)
health gate (DeepSeek ~10-20min cold start) → (next)
train/loss over steps           → green
```
