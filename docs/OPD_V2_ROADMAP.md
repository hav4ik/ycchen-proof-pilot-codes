# OPD v2 on cu128/B200 — roadmap & readiness (start here)

Goal: run Yi-Chia Chen's OPD v2 (on-policy distillation, Olmo3-32B student) on **CUDA 12.8 / B200**,
staying **faithful** to her code — the only intended deviations are (1) cu128 packaging and (2) the FA2
attention-sink for B200 (no FA3). Everything lives on fork branch **`opd/b200-cu128`**, which is
**her latest `main` + only that delta** (sub-agent-audited; core `training/opd_v2/src/` byte-identical
to hers). Two small additive config knobs exist for the smaller tests (`OPD_HID_DIM`, `TEACHER_PATH`),
both defaulting to her exact values.

## The image: `ycchen-opd:cu128`

One image, two venvs. Base `chankhavu/olmo3-olmocore:cu128-fa2-sink`.
- **trainer + orchestrator** (base py, torch 2.10+cu128): transformers 5.9.0, `olmo3_sink_fa2` (stock FA2 +
  post-correction sink), liger. flash-attn is a **stock multi-arch wheel (sm_90 H200 + sm_100 B200)** —
  build it first with `docker/cu128/build_fa2_wheel.sh` (the base ships Blackwell-only; no pre-built wheel
  exists for cu128+torch2.10). nvcc is baked in.
- **rollout + teacher** (`/opt/venv/serve`, sglang 0.5.14): her patched sources re-anchored to 0.5.14;
  rollout `--attention-backend triton` (sink-correct on B200); teacher DeepSeek-V4-Flash hidden-extract.

To (re)build: `OUT=docker/cu128/wheels bash docker/cu128/build_fa2_wheel.sh` then
`DOCKER_BUILDKIT=1 docker build -f docker/cu128/Dockerfile.ycchen-opd -t ycchen-opd:cu128 .`

## The three run configs

| config | GPUs | teacher | launcher + preset | purpose |
|---|---|---|---|---|
| **4×H200 Olmo3** | 4 (1:1:2) | Olmo3-32B (self or 2nd ckpt) | `run_1node.sh` + `env_4gpu_olmo3.sh` | cheapest loop check, **no DeepSeek download**; real JSD if two ckpts |
| **8×H200 smoke** | 8 (4:2:2) | DeepSeek-V4-Flash | `run_1node.sh` + `env_1node_smoke.sh` | the **main integration test** before B200 (exact production teacher path) |
| **64×B200 run** | 64 = 8 nodes | DeepSeek-V4-Flash | `run_mn_cu128.sh` + `env_v33_b200.sh` | the production run, her exact V33 |

All three use the **same code path** (olmo3_sink_fa2 trainer + triton rollout); they differ only in
scale/topology/teacher. Runbooks: [4×H200](OPD_V2_4GPU_OLMO3.md) · [8×H200](OPD_V2_H200_SMOKE.md).

## Readiness

### Path A — 8×H200 smoke → **software-ready**
Everything is built: image, `run_1node.sh` + `env_1node_smoke.sh` (4:2:2, DeepSeek TP4 / rollout TP2 /
trainer world-2, 40k, single_round, `CHECKPOINT_EVERY=0`), full step-by-step runbook. **Remaining is
operational (your side):** push image → pull on node; download student (~65 GB) + DeepSeek-V4-Flash
(hundreds of GB); run. `marlin` MoE backend works on Hopper — no B200 change here. The shakeout is the validation.

### Path B — 64×B200 run → **software-ready except one B200 item**
`run_mn_cu128.sh` is a faithful slurm port of her `run_mn.sh` (same topology 1 teacher + 4 rollout +
3 trainer / world 24, ports, health gate); `env_v33_b200.sh` is her exact V33 config + the two cu128
deltas. **Open items:**
1. **DeepSeek teacher MoE on sm_100 — de-risked, not a blocker.** `marlin` is Hopper-only, but
   `run_teacher.sh` now defaults `MOE_BACKEND=auto` so sglang picks per hardware (marlin on Hopper = her
   V33; a Blackwell backend on B200). Per sglang **issue #23743**, *"a full DeepSeek-V4-Flash TP=4 server
   can run on one GB200 node"* — the validated backend there is `flashinfer_mxfp4` (available in our
   0.5.14), and a known FlashMLA mixed decode+prefill crash has the workaround `--max-prefill-tokens 8192`.
   Both are documented overrides (`MOE_BACKEND=flashinfer_mxfp4`, `MAX_PREFILL_TOKENS=8192`) to use only
   if auto's pick misbehaves on the first B200 run.
2. **Multi-node Beaker wiring** — `run_mn_cu128.sh` is slurm-native; wrap in Beaker's multi-node
   primitives at run time (mechanical).
3. B200 allocation + first sm_100 kernel exercise (also the moment to confirm the teacher MoE backend).

## Prerequisites (both H200/B200 paths)

- **FA2 wheel built** (`build_fa2_wheel.sh`) before the image build — it's gitignored (130 MB).
- **Models** on the node/FS: student **deploy-format** Olmo3-32B (`chankhavu/yccchen-olmo3-deploy` —
  verified loadable by both trainer and rollout; the deploy config doubles as the training config) +
  DeepSeek-V4-Flash (for the DeepSeek-teacher paths).
- **Disk**: run `RUN_DIR` peak ≈ 130 GB for a smoke (checkpointing off) → ~1.8 TB for the production run
  (fp32 DCP checkpoints dominate; the `rollouts/` dump grows unbounded — cap with `ROLLOUT_DUMP=0`).
  See [OPD_V2_H200_SMOKE.md](OPD_V2_H200_SMOKE.md) / the disk notes.

## Sequence

```
(optional) 4×H200 Olmo3 loop check  ─▶  8×H200 smoke (main gate)  ─▶  64×B200 run
   no DeepSeek needed                    validates the DeepSeek path      new unknowns: MoE backend + sm_100
```

Run the 8×H200 smoke first; if green, the only genuinely-new risks on B200 are the MoE backend flag and
the sm_100 kernels themselves.

## One unverified assumption (4×H200 Olmo3 path only)

The Olmo3 teacher relies on the baked `olmo2.py` surfacing hidden states via sglang's
`--enable-return-hidden-states` (DeepSeek needed `patch_dsv4`; Olmo3 *should* work through the generic
path). If teacher `/score` returns empty hidden on the first run, that's the fix — model-side, not config.
The 8×H200 / B200 DeepSeek path does not depend on this.

## Doc index
- [OPD_V2_ALGORITHM.md](OPD_V2_ALGORITHM.md) — how her algorithm works (env, rollout, JSD, weight-sync).
- [OPD_V2_CONFIG_REFERENCE.md](OPD_V2_CONFIG_REFERENCE.md) — every knob + her best OPD-32B V33 values.
- [OPD_V2_H200_SMOKE.md](OPD_V2_H200_SMOKE.md) · [OPD_V2_4GPU_OLMO3.md](OPD_V2_4GPU_OLMO3.md) — runbooks.
- [OPD_V2_PARITY_STATUS.md](OPD_V2_PARITY_STATUS.md) — cu13↔0.5.14 parity + faithfulness audit results.
- [../docker/cu128/README.md](../docker/cu128/README.md) — image build + launchers.
