# OPD v2 — FlashInfer attention-sink: evaluation & decision

**Decision (2026-07-13): production stays on `triton`.** The native FlashInfer attention-sink path is
integrated, GPU-correctness-validated, and kept as a **proven fallback**, but the measured gain (~9% on
bf16 KV) does not justify FlashInfer's fp8-native attention deviation on a precision-sensitive model where
Yi-Chia's validated production backend is `triton`.

## What it is

- Branch **`opd/flashinfer-sink`**; image **`chankhavu/ycchen-opd:cu128-flashinfer-sink`** (`sha256:30994b4d…`).
- Adds a native FlashInfer sink path (re-derived onto sglang 0.5.14) + a **flashinfer 0.6.12 → 0.6.14** bump
  (the sink JIT APIs `get_batch_prefill_attention_sink_uri` / `attention_sink_decl` are 0.6.14-only).
- **Opt-in** — the rollout still defaults to `--attention-backend triton`; select FlashInfer with
  `ATTENTION_BACKEND=flashinfer`. So this image behaves identically to production unless you switch.
- **BF16 sinks preserved** (per directive): the sink `Parameter` stays bf16 like the triton path; the
  FlashInfer kernel casts the tiny per-head tensor to fp32 at the call site.

## Correctness — VALIDATED ✅

`docker/cu128/tests/flashinfer_sink/test_flashinfer_attention_sink.py` → **100 passed** on a live H200:
FlashInfer ≈ eager ≈ triton across prefill/extend/decode × `window_left{-1,4}` × sink values × GQA ×
KV `{bf16, fp8_e4m3}` × non-unit fp8 scales, **including the cuda-graph sink-reload case** — which was the
one residual risk of the bf16-param + in-kernel-cast approach. It passed, so the bf16 sink choice is correct
under weight reload. Tolerances vs the **fp32 eager reference**: `3e-2` (bf16 KV), `8e-2` (fp8 KV).

Running the tests needs two prereqs (both in the branch's `tests/flashinfer_sink/README.md`): `pytest` in
the serve venv, and the **forward-compat serve env** loaded (`LD_LIBRARY_PATH=/opt/cuda13-compat` +
`CUDA_HOME=/usr/local/cuda-13.0`) — otherwise the cu130 serve venv falls back to CPU and the CUDA-guarded
sink wrapper isn't importable.

## Throughput A/B

Measured **~525 tok/s (flashinfer) vs ~481 (triton)** ≈ **+9%** — BUT on **bf16 KV** (the A/B left
`KV_CACHE_DTYPE` unset → sglang default). Production runs **fp8 KV** (`KV_CACHE_DTYPE=fp8_e4m3`), so re-run
with `-e KV_CACHE_DTYPE=fp8_e4m3` on both backends for the production-representative number (that also
exercises the exact fp8 path discussed below).

## The fp8-precision nuance (why the decision is conservative)

- **KV *storage* dtype is a server flag (`--kv-cache-dtype`), identical for both backends** —
  `run_rollout.sh:89` passes it regardless of `ATTENTION_BACKEND`. It is NOT "triton stores bf16, FlashInfer
  stores fp8"; both store whatever `KV_CACHE_DTYPE` says. The divergence is purely in the **compute step**.
- On fp8 KV the two backends compute differently (verified in `tests/flashinfer_sink/kernel_test_utils.py`):
  - **triton** upcasts fp8 KV → bf16 first (`k.to(q.dtype)`, q.dtype=bf16) → **attention math in bf16**.
  - **FlashInfer** passes fp8 KV in with `k_scale`/`v_scale` → **fp8-native** (fp8 matmul *inputs* + fp32
    accumulate + fp32 softmax — not "everything in fp8").
- The differential test **bounds the delta**: FlashInfer ≈ triton ≈ fp32-reference within **8e-2** on fp8 KV
  (with a direct flashinfer-vs-triton assert). So FlashInfer is not *systematically* worse — the deviation is
  bounded, not free.

## Decision & rationale

**Keep `triton` for the production/Beaker run.** Yi-Chia's validated production is triton + fp8 KV;
FlashInfer's fp8-native attention is a bounded-but-real deviation from that exact path, and ~9% (bf16) does
not justify it on a precision-sensitive model. FlashInfer stays a **proven fallback**: flip
`ATTENTION_BACKEND=flashinfer` if rollout throughput ever becomes the actual bottleneck, and re-validate the
production checklist (tests 1–4 in [OPD_V2_H200_SMOKE.md](OPD_V2_H200_SMOKE.md)) on that image first.

On-policy note: switching would **not bias training** — the student trains on its *own* rollouts, so a small
rollout-attention numeric difference only shifts which tokens are sampled, not the objective. But matching the
deploy/validated backend (triton) is the correct default for the real run.

## How to re-evaluate later

1. **Correctness:** `cd docker/cu128/tests/flashinfer_sink && <forward-compat env> && $SERVE_PY -m pytest -q
   test_flashinfer_attention_sink.py` (from branch `opd/flashinfer-sink` / the flashinfer image).
2. **Throughput (production-representative):** launch the rollout twice — `--attention-backend triton` vs
   `flashinfer`, **both with `KV_CACHE_DTYPE=fp8_e4m3`** — benchmark identical `/generate` load, compare the
   steady-state `gen throughput (token/s)` in each `rollout.log`.
3. Only adopt if the fp8-KV gain is **large enough to matter** for the run, not a few percent.
