#!/usr/bin/env bash
# OPD v2 ROLLOUT server (student, fp8 flash_rl) — in-container, Blackwell/B200.
# De-apptainer'd port of ycchen training/opd_v2/flash_rl/run_rollout_fp8.sh:
#   * no `apptainer exec $SIF` (we run in the image's serve venv)
#   * patches (olmo2_sink model, flash_rl loader, SWA model_config) are BAKED into the venv at build
#     time, so no --bind overlays here
#   * this fork uses FlashInfer's dedicated attention-sink JIT; Blackwell must pass
#     the registered differential tests before selecting it over the Triton default.
#
#   CUDA_VISIBLE_DEVICES=1 ./run_rollout.sh --port 8201
set -euo pipefail

# --- serve venv is CUDA-13 (sglang 0.5.14): use the CUDA-13 nvcc/CCCL for its runtime JIT kernels -----
# The base image defaults CUDA_HOME to cu128, but sglang JIT-compiles kernels that use CUDA-13 cuda::ptx
# intrinsics -> point it at the CUDA-13 toolkit baked in the image (the base/trainer venv stays cu128).
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:${PATH}

# --- CUDA-13 forward-compat (see Dockerfile cuda13compat stage) --------------------------------------
# sglang 0.5.14 is a CUDA-13 build and can't be pinned to cu128 (sgl-project/sglang#25069). If this
# node's driver predates CUDA 13, load the baked forward-compat libcuda so the serve stack runs on
# CUDA-12.x datacenter drivers (>=525). On a CUDA-13 driver this is skipped (native libcuda is used).
if [ -d /opt/cuda13-compat ]; then
  _cc=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]\{1,\}\).*/\1/p' | head -1)
  if [ -n "${_cc}" ] && [ "${_cc}" -lt 13 ] 2>/dev/null; then
    export LD_LIBRARY_PATH="/opt/cuda13-compat:${LD_LIBRARY_PATH:-}"
    echo "[cuda13-compat] node driver CUDA ${_cc}.x < 13 -> forward-compat libcuda enabled" >&2
  fi
fi
SERVE_PY="${SERVE_PY:-/opt/venv/serve/bin/python}"
MODEL="${MODEL:?set MODEL=<student deploy dir>}"
PORT=8200; TP="${TP:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
while [ $# -gt 0 ]; do case "$1" in
  --port) PORT=$2; shift 2;;
  --tp) TP=$2; shift 2;;
  --attention-backend) ATTENTION_BACKEND=$2; shift 2;;
  *) shift;;
esac; done

case "$ATTENTION_BACKEND" in
  triton|flashinfer) ;;
  *) echo "unsupported sink attention backend: $ATTENTION_BACKEND" >&2; exit 2;;
esac

if [ "$ATTENTION_BACKEND" = flashinfer ]; then
  "$SERVE_PY" - <<'PY'
from flashinfer import BatchAttentionWithAttentionSinkWrapper
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    _run_flashinfer_paged_with_sinks,
)
from sglang.srt.model_loader.loader import QuantizedRLModelLoader
from sglang.srt.models.olmo2 import Olmo3SinkForCausalLM

assert BatchAttentionWithAttentionSinkWrapper is not None
assert FlashInferAttnBackend is not None
assert _run_flashinfer_paged_with_sinks is not None
assert hasattr(QuantizedRLModelLoader, "_validate_attention_sink_checkpoint")
assert hasattr(Olmo3SinkForCausalLM, "validate_loaded_attention_sinks")
print("[rollout] verified native FlashInfer attention-sink serving support")
PY
fi

EXTRA_ARGS=()
[ -n "${CONTEXT_LEN:-}" ]   && EXTRA_ARGS+=(--context-length "$CONTEXT_LEN")
[ -n "${KV_CACHE_DTYPE:-}" ] && EXTRA_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")   # fp8_e4m3 for long ctx
[ -n "${SWA_RATIO:-}" ]     && EXTRA_ARGS+=(--swa-full-tokens-ratio "$SWA_RATIO")
[ "${TP:-1}" -gt 1 ]       && EXTRA_ARGS+=(--disable-custom-all-reduce)            # two TP-groups/node crash guard
# Triton-only fallback tuning.
[ "$ATTENTION_BACKEND" = triton ] && [ -n "${KV_SPLITS:-}" ] && \
  EXTRA_ARGS+=(--triton-attention-num-kv-splits "$KV_SPLITS")

PARSER_ARGS=(--reasoning-parser deepseek-r1 --tool-call-parser deepseekv4)
# Default matches Yi-Chia's run_rollout_fp8.sh (SKIP_TOKENIZER_INIT=0 -> tokenizer + parsers on); her
# production run_mn.sh/sbatch rely on this default. Opt into token-only with SKIP_TOKENIZER_INIT=1.
[ "${SKIP_TOKENIZER_INIT:-0}" = "1" ] && PARSER_ARGS=(--skip-tokenizer-init)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$SERVE_PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size "$TP" --host 0.0.0.0 --port "$PORT" \
  --attention-backend "$ATTENTION_BACKEND" \
  --page-size 1 \
  --quantization fp8 --load-format flash_rl \
  --mem-fraction-static "${MEMFRAC:-0.85}" \
  --max-running-requests "${MAXRUN:-10}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-${MAXRUN:-10}}" \
  --chunked-prefill-size "${CHUNKED_PREFILL:-4096}" \
  --disable-radix-cache \
  "${EXTRA_ARGS[@]}" \
  "${PARSER_ARGS[@]}"
