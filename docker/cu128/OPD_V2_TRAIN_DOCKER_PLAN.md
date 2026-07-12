# OPD v2 training container — B200 / CUDA 12.8 build plan

Single Docker image running the **full OPD v2 on-policy-distillation loop** (teacher · rollout ·
trainer · orchestrator) on **NVIDIA B200 (Blackwell, sm_100)**, hard-pinned to **CUDA 12.8**, on the
AI2 (Beaker + Weka) cluster.

Source studied: `ycchen-proof-pilot-codes/training/opd_v2/` + `olmo3_sink/` + parent `DOCKER_PLAN.md`
(the sm120/cu130 vast **serving** study — reused where it applies).

---

## 1. Constraints

| Constraint | Consequence |
|---|---|
| **CUDA 12.8 hard ceiling** | It is also the *floor*: Blackwell sm_100 needs CUDA ≥ 12.8. On cu128, torch tops out at **2.11** (per the repo's own pyproject note). Target `torch==2.11.0+cu128`. |
| **B200 / sm_100 → no FA3** | FA3 kernels are **Hopper-only** (`flash-attention/hopper/`). The production `attn="olmo3_sink_fa3"` path is dead on B200. Replace with `flash_attention_2` (primary) / `flex_attention` (fallback). |
| **AI2 = Beaker docker, same image on every node** | One image, launched per-role as separate Beaker tasks sharing a Weka `run_dir`. No apptainer. |
| **Weka shared FS** | Native fit — OPD's hidden-state transport (`hidden_store.py`) already assumes WekaFS `/work` with atomic-rename. |

---

## 2. Why this is tractable (the findings that de-risk it)

1. **FA3 is only the *fast path*, not the only path.** `modeling_olmo3_sink.py:14-16` states the
   sink (`s_aux`) works with **`eager`, `flash_attention_2`, `flash_attention_3`, `flex_attention`**
   (only `sdpa` is unsupported). The model already dispatches `s_aux=self.sinks` into transformers'
   attention interface (`modeling_olmo3_sink.py:168-189`). **We change a config flag and validate — we
   do not port a CUDA kernel.**
2. **A correct hardware-agnostic reference already exists:** `eager_attention_forward_with_sink`
   (`modeling_olmo3_sink.py:66-101`) — gpt-oss-style sink column in a plain softmax. Ground truth for
   parity tests.
3. **`register.py` already tolerates a missing FA3 build** — `flash_attn_interface` import is lazy;
   without it the model imports and runs on non-FA3 backends. So dropping the patched-FA3 wheel is a
   supported configuration, not a hack.
4. **Serving-side sink on Blackwell is already solved** (parent `DOCKER_PLAN.md`): `fa3` is Hopper-only,
   **flashinfer silently drops the sink → use `--attention-backend triton`**; the sink math rides
   sglang's *stock* triton extend+decode kernels via the `olmo2_sink.py` model patch. Same rule for
   sm_100 as for sm_120.
5. **The v2 architecture is transport/discovery-clean** — config SSOT (`config.py`), FS hidden
   transport, `trainer_endpoint.json` self-registration, torchrun HSDP mesh auto-derivation
   (`stage1_v2/src/train.py:183`). None of it needs porting.

---

## 3. Image architecture — single image, isolated venvs

One image; two Python virtualenvs isolate the two conflicting dependency worlds (training torch vs
sglang's pinned torch/flashinfer/sgl_kernel). Built with **`uv`** for reproducibility (repo ships `uv.lock`).

```
FROM nvidia/cuda:12.8.<patch>-cudnn-devel-ubuntu22.04   # devel: nvcc for FA2 source build + JIT
/opt/venv/train    # trainer + orchestrator
/opt/venv/serve    # teacher + rollout (sglang)
/opt/opd/repo      # (bind-mounted from Weka at runtime; NOT copied — it changes)
/opt/opd/bin/*.sh  # de-apptainer'd launchers, each activates the right venv
```

| Role | venv | Entry |
|---|---|---|
| trainer | `train` | `torchrun … -m opd_v2.trainer.service --run-dir <weka>/run` |
| orchestrator | `train` (CPU-only; torch-free import path) | `python -m opd_v2.orchestrator --run-dir <weka>/run` |
| teacher | `serve` | `python -m sglang.launch_server …` (DeepSeek-V4-Flash, hidden-extract patches) |
| rollout | `serve` | `python -m sglang.launch_server …` (student fp8, `olmo2_sink` patch) |

Rationale for multi-venv over two images: honors AI2's "same container" model, keeps one artifact to
stage on Weka/Beaker, and still prevents the sglang↔training torch-version fight.

---

## 4. Pinned version matrix (to be locked in Phase 0)

**`train` venv** (Blackwell training):
| Package | Pin | Note |
|---|---|---|
| torch | `2.11.0+cu128` | max torch on cu128; has FSDP2, FlexAttention, sm_100 |
| triton | bundled w/ torch 2.11 | sm_100 capable; drives FlexAttention + liger |
| transformers | `>=4.57` | has `s_aux` FA2/Flex support (from gpt-oss) |
| flash-attn (FA2) | `>=2.7.x`, sm_100 build | **may need source build** vs cu128/torch2.11 → FlexAttention fallback |
| liger-kernel | `>=0.8` | triton, sm_100 |
| + repo deps | from `pyproject.toml` | minus the cu126 torch source pin (override to cu128) |

**`serve` venv** (Blackwell sglang) — versions TBD in Phase 3; the patches are content-pinned, so the
sglang version is chosen to (a) support DeepSeek-V4 MLA+MoE on sm_100 and (b) accept the hidden-extract
+ `olmo2_sink` patches (regenerate via `REGEN_LOADER`/REPATCH if the line anchors moved).

---

## 5. Phased plan

### Phase 0 — Lock the training stack (½ day)
- Create `/opt/venv/train` with `uv`; override the pyproject `pytorch-cu126` source → **cu128 / torch 2.11**.
- Wire sibling libs (`_common`, `_vendor_opd`, `stage1_v2/src`, `distill_gen/math_3r`) via the repo's
  `sys.path`-from-`__file__` mechanism (repo mounted from Weka; not pip-installed).
- Sanity: `import torch; torch.cuda.get_device_capability()` == (10, 0); FSDP2 import; `register_olmo3_sink()`
  imports cleanly *without* `flash_attn_interface`.
- **Deliverable:** reproducible train venv; **Accept:** repo imports + a CPU eager forward runs.

### Phase 1 — B200 training attention (CORE RISK, 1–2 days)
Retarget `TrainerCfg.attn` (`config.py`) / `ATTN_IMPL` (`stage1_v2/src/train.py:68`) off `olmo3_sink_fa3`.
- **Primary `flash_attention_2`:** obtain/build a **Blackwell FA2 wheel** (flash-attn ≥2.7.x, sm_100,
  vs cu128/torch2.11). transformers passes `s_aux`.
- **Fallback `flex_attention`:** built-in torch, no external wheel; guaranteed sm_100.
- **Validate both vs the `eager` sink reference:**
  1. forward parity (logits) at short + long seq;
  2. gradient parity — dq/dk/dv **and** `dsink` (the sink param is trainable; `fa3_sink.py` gives the
     closed-form dsink to cross-check);
  3. **varlen document packing** — cu_seqlens / position_ids isolation (trainer packs whole un-windowed
     ~64k trajectories, `micro_batch_tokens=65536`);
  4. grad-checkpointing (`use_reentrant=False`) + `torch.compile` (no graph-break regressions).
- **Deliverable:** `attn` value that trains correctly on B200. **Accept:** parity < tol vs eager on all 4;
  FA2 chosen if the wheel builds and beats Flex at 64k, else Flex.

### Phase 2 — Trainer image + smoke (½ day)
- Dockerfile: base + both venvs; **strip every `apptainer exec --nv $SIF`**; run `torchrun` directly.
- Single-B200 smoke: `smoke_test_opd.py` + `opd_v2/tests/`; then 2-GPU FSDP2 shard check (assert first
  param is a `DTensor` — `core.py:271`).
- **Accept:** one real `train_step` on synthetic hidden + a weight `/save` round-trip.

### Phase 3 — Serve venv for B200 (higher uncertainty, 2–3 days)
- `serve` venv: sglang with sm_100 kernels on cu128 (sgl_kernel + flashinfer Blackwell builds).
- **Rollout (student):** apply `olmo2_sink.py` patch; `--attention-backend triton` (sink correctness);
  fp8 `--load-format flash_rl` + patched `loader.py`/`model_config.py`; validate
  `update_weights_from_disk` bit-exact reload on Blackwell (Hopper note "e4m3 keeps FA3" is void here —
  re-measure KV dtype × backend).
- **Teacher (DeepSeek-V4-Flash):** apply hidden-extract patches (`deepseek_v4.py`, `scheduler.py`,
  `http_server.py` + the v2 `/score` FS-write); `--enable-return-hidden-states`; **swap
  `--moe-runner-backend marlin` → a Blackwell MoE backend** (cutlass/triton/flashinfer — benchmark);
  pick an sm_100 MLA attention backend.
- Re-pin patches to the chosen sglang (`REGEN_LOADER=1` / REPATCH if anchors moved).
- **Accept:** teacher `/score` writes a valid hidden file (`read_hidden` header check) + rollout serves
  and reloads weights.

### Phase 4 — Beaker multi-node wiring (~1 day)
- The missing `run_mn`: per-role Beaker tasks sharing a Weka `run_dir`; rendezvous via
  `ROLLOUT_URLS`/`TEACHER_URLS` (`make_config.py`) + `trainer_endpoint.json` (`service.py:118`).
- Trainer task = only true cross-node collective: `torchrun` with Beaker leader → `MASTER_ADDR`,
  `LOCAL_WORLD_SIZE` set so HSDP mesh derives (`train.py:197`).
- Cross-node landmines already documented: `--disable-custom-all-reduce` for >1 TP-group/node
  (`run_rollout_fp8.sh:62`).
- **Accept:** 2-node end-to-end: N steps + weight sync + durable checkpoint.

---

## 6. De-apptainer checklist (concrete)
`run_teacher_fs.sh` and `flash_rl/run_rollout_fp8.sh` both `exec apptainer exec --nv … $SIF python3 …`
with `--bind` overlays. In-container equivalent:
- Drop the `apptainer exec --nv $SIF` wrapper → call the `serve` venv python directly.
- Replace each `--bind SRC:DST` by **baking** the patched file over the installed sglang at image-build
  time (the scripts already encode every `SRC:DST` pair, e.g. `loader.py:$SGL/srt/model_loader/loader.py`).
- Keep the `--env` values as container ENV / launcher exports.

---

## 7. Risk register
1. **Phase 3 serve (teacher DeepSeek-V4-Flash on B200 + patch re-pin)** — highest uncertainty.
2. **FA2 sm_100 wheel build** — mitigated by FlexAttention fallback (no wheel).
3. **Attention numerical parity** — mitigated by the built-in eager reference.
4. **Blackwell MoE backend** for the teacher (marlin unavailable/suboptimal) — benchmark cutlass/triton.

## 8. Open items to confirm during build
- Exact cu128 patch base image + whether cudnn-devel suffices for the FA2 source build.
- Whether a prebuilt flash-attn sm_100 wheel exists for torch2.11+cu128 (else build from source).
- Which sglang version simultaneously supports DeepSeek-V4 on sm_100 **and** the OPD patch anchors.
- Rollout KV-cache dtype × attention-backend that preserves sink correctness on Blackwell.

## 9. Acceptance (end state)
Single `opd:cu128-b200` image; 2-node Beaker run of the full loop (teacher `/score` → buffer →
`/train_step` → weight sync → durable checkpoint) producing decreasing reverse-KL, with the trainer on
a validated non-FA3 sink attention.
