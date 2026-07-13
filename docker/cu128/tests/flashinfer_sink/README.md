# FlashInfer attention-sink tests (OPD serve, sglang 0.5.14)

These tests come from the fork `hav4ik/sglang @ codex/flashinfer-attention-sink` and
were re-homed here for the OPD serve overlay
(`docker/cu128/opd_serve/sglang_patches/`). They exercise the FlashInfer dedicated
attention-sink JIT path that `run_rollout.sh --attention-backend flashinfer` selects
for `Olmo3SinkForCausalLM` checkpoints.

## Requirements

- **1 NVIDIA GPU** (all tests are CUDA-only; the differential kernel test hard-skips
  when `torch.cuda.is_available()` is false, and every module `pytest.importorskip`s
  `flashinfer`).
- The **serve venv** at `/opt/venv/serve` with the baked sglang overlays, and
  **flashinfer 0.6.14** (`flashinfer-python`/`flashinfer-cubin`/`flashinfer-jit-cache`,
  bumped by `Dockerfile.ycchen-opd`). Earlier flashinfer (0.6.12) lacks
  `get_batch_prefill_attention_sink_uri` / `attention_sink_decl` and the tests error at
  import.
- **pytest** in the serve venv: `uv pip install --python /opt/venv/serve/bin/python pytest`.
- **fp8 KV cases skip below sm89** — the fp8_e4m3fn parametrizations and
  `test_fp8_kv_attention_sinks_apply_nonunit_scales` require compute capability >= 8.9
  (Ada/Hopper/Blackwell); on older GPUs they skip.

## Run commands

Run from **inside this directory** so the shared `kernel_test_utils.py` helper is
importable (it is a flat module, not a package):

```bash
cd docker/cu128/tests/flashinfer_sink
SERVE_PY=/opt/venv/serve/bin/python

# REQUIRED on a pre-CUDA-13 driver (e.g. H200/570): the cu130 serve venv needs the forward-compat
# libcuda + CUDA-13 toolkit loaded to see the GPU (run_{rollout,teacher}.sh do this at launch; a
# standalone test must set it too). WITHOUT it, torch rolls back to CPU ("Triton is not supported on
# current platform, roll back to CPU") -> flashinfer reports unavailable -> the CUDA-guarded sink
# wrapper class SGLangBatchAttentionWithAttentionSinkWrapper is never defined -> ImportError at collect.
# On a native CUDA-13 driver (B200) this is a harmless no-op.
export LD_LIBRARY_PATH=/opt/cuda13-compat:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH

# Core numerical validation: FlashInfer sink kernel vs Triton vs eager reference
# (prefill / extend / decode, bf16 + fp8 KV, sliding-window and full attention).
"$SERVE_PY" -m pytest -q test_flashinfer_attention_sink.py

# Unit tests for the ported flashinfer_backend + model_config sink hunks.
"$SERVE_PY" -m pytest -q test_flashinfer_attention_sinks.py

# Unit tests for the sink checkpoint-validation hooks in olmo2 + the loader.
"$SERVE_PY" -m pytest -q test_flash_rl_attention_sinks.py
```

## What passes here vs. what is coupled to the fork's own rewrite

Our overlays graft **only** the flashinfer-sink-specific additions onto Yi-Chia Chen's
existing `olmo2.py` / flash_rl `loader.py`. We deliberately did **not** adopt the fork's
re-implementation of code that is already hers (its `get_parallel()` API, its
`_should_quantize_weight` helper, its stricter `_apply_scale_update(validate_only=...)`
scale logic, and its staging rollback in `rebinding_and_load_weights`). Several fork
tests assert that fork-only behavior and therefore **do not** map cleanly onto our
faithful-to-YCC overlays. Treat the list below as the source of truth for pass/fail:

### Expected to pass (validate THIS integration)

- `test_flashinfer_attention_sink.py` — **the primary signal.** Differential
  FlashInfer-sink vs Triton vs eager across prefill/extend/decode, bf16/fp8 KV,
  window/full. Exercises our ported `SGLangBatchAttentionWithAttentionSinkWrapper` and
  `_run_flashinfer_paged_with_sinks`.
- `test_flashinfer_attention_sinks.py`: `test_flashinfer_native_sink_run_sets_runtime_options`,
  `test_flashinfer_sink_jit_requires_float32_sinks`, `test_flashinfer_sink_requires_page_size_one`,
  `test_sink_jit_cache_identity_includes_kv_dtype`, `test_olmo3_sink_config_detection`
  (our `_detect_attention_sinks` hunk), `test_yccchen_deploy_config_preserves_sink_architecture`,
  `test_sink_models_disable_ragged_prefill`, `test_sink_prefill_indices_updater_reads_backend_flag`.
- `test_flash_rl_attention_sinks.py`: `test_flash_rl_requires_every_sink_on_disk_reload`,
  `test_flash_rl_requires_olmo_norms_with_sink_reload`, `test_initial_load_rejects_missing_sink_tensor`,
  `test_olmo3_expected_checkpoint_contract_includes_all_projections`,
  `test_flash_rl_rejects_missing_projection_before_staging`,
  `test_flash_rl_does_not_nest_reload_proxy` (our preserved nested-proxy guard).

### Auto-skipped (dependency intentionally not ported)

- `test_yccchen_rope_theta_accepts_transformers_5_schema` — skips: our `olmo2.py`
  overlay inlines rope-theta selection and has no standalone `get_olmo_rope_theta`.
- `test_gpt_oss_defaults_to_flashinfer_when_available` — `pytest.importorskip`s the
  absent `sglang.srt.arg_groups.overrides` module (not present in stock 0.5.14).

### Expected to FAIL / ERROR against our overlays — need manual adaptation

These target the fork's loader/olmo2 internals, which we kept as Yi-Chia's:

- `test_olmo3_sink_weight_loads_target_tp_shard`,
  `test_olmo3_sink_weight_rejects_wrong_head_count`,
  `test_flash_rl_reloads_sink_values_without_reallocating_parameter` — patch
  `sglang.srt.models.olmo2.get_parallel`. Our overlay shards sinks with
  `get_tensor_model_parallel_rank()` instead. To validate our code, repoint the patch
  to `sglang.srt.models.olmo2.get_tensor_model_parallel_rank` returning the desired
  rank (e.g. `return_value=3` for the `[15:20]` shard assertion). Also, our sink branch
  does not raise the fork's `expected [40]` `ValueError`, so
  `test_olmo3_sink_weight_rejects_wrong_head_count` has no equivalent here.
- `test_flash_rl_keeps_one_dimensional_sink_in_bfloat16` — calls
  `QuantizedRLModelLoader._should_quantize_weight`, which we did not port (our loader
  uses the inline `weight.dim() >= 2` guard). No equivalent public helper.
- `test_flash_rl_failed_reload_preserves_sink_value_and_pointer` — asserts the fork's
  try/except rollback in `rebinding_and_load_weights`; our overlay keeps Yi-Chia's
  staging (no rollback), so this does not apply.
- `test_flash_rl_rejects_missing_stacked_scale_before_commit`,
  `test_flash_rl_applies_gqa_qkv_scales_with_unequal_shard_rows` — assume the fork's
  `_apply_scale_update(validate_only=...)` strict-scale rewrite and patch
  `loader.get_parallel`; our `_apply_scale_update` is Yi-Chia's warn-and-continue
  version keyed on `get_tensor_model_parallel_rank()`.

`register_cuda_ci` imports are wrapped in a try/except no-op fallback in every file so
collection does not depend on the CI-only `sglang.test.ci` module shipping in the wheel.

> All of the above is best-effort pending GPU validation. The kernel differential test
> is the load-bearing check that the ported FlashInfer sink path is numerically correct.
