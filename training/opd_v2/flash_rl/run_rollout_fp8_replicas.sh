#!/usr/bin/env bash
# Launch independent Flash-RL rollout processes.
#
# Each replica is an independent SGLang process with its own TP group. This is
# This is a fallback for deployments that cannot use SGLang native DP. Prefer
# run_rollout_fp8.sh --tp 1 --dp 8 on a single eight-GPU node: SGLang's
# DataParallelController owns the eight replicas behind one endpoint.
#
# Example: one eight-GPU node, eight TP=1 rollout replicas:
#   GPU_IDS=0,1,2,3,4,5,6,7 ADVERTISE_HOST=node-a \
#     ./run_rollout_fp8_replicas.sh --replicas 8 --tp 1 --port-base 8200
#
# Run once per policy node, concatenate each node's rollout_urls.txt into
# ROLLOUT_URLS, and set TARGET_INFLIGHT <= replicas * MAXRUN across all nodes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT_BASE="${PORT_BASE:-8200}"
REPLICAS="${ROLLOUT_REPLICAS:-1}"
TP="${ROLLOUT_TP:-1}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
LOG_DIR="${ROLLOUT_LOG_DIR:-$PWD/rollout-replica-logs}"
ADVERTISE_HOST="${ADVERTISE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
URLS_FILE="${ROLLOUT_URLS_FILE:-}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_rollout_fp8_replicas.sh [options]

Options:
  --replicas N          Number of independent rollout processes (default: ROLLOUT_REPLICAS or 1).
  --tp N                GPUs per replica / SGLang TP group (default: ROLLOUT_TP or 1).
  --port-base PORT      First server port (default: PORT_BASE or 8200).
  --gpu-ids CSV         Physical GPU ids, grouped contiguously per replica.
  --advertise-host HOST Hostname/IP written to the generated endpoint list.
  --log-dir DIR         Per-replica server logs and rollout_urls.txt directory.
  --urls-file PATH      Explicit endpoint-list output path.
  --dry-run             Print assignments and generated URLs without starting servers.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --replicas) REPLICAS="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --advertise-host) ADVERTISE_HOST="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --urls-file) URLS_FILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Resolve this after parsing --log-dir so the two defaults remain coupled.
URLS_FILE="${URLS_FILE:-$LOG_DIR/rollout_urls.txt}"

if ! [[ "$REPLICAS" =~ ^[1-9][0-9]*$ && "$TP" =~ ^[1-9][0-9]*$ && "$PORT_BASE" =~ ^[0-9]+$ ]]; then
  echo "replicas, tp, and port-base must be positive integers" >&2
  exit 2
fi
if [ "$REPLICAS" -gt 1 ] && [ "${REGEN_LOADER:-0}" = "1" ]; then
  echo "REGEN_LOADER=1 is unsafe with multiple replicas; regenerate the patch once before launching replicas" >&2
  exit 2
fi

required_gpus=$((REPLICAS * TP))
if [ -z "$GPU_IDS" ]; then
  GPU_IDS="$(seq -s, 0 $((required_gpus - 1)))"
fi
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
if [ "${#gpu_array[@]}" -ne "$required_gpus" ]; then
  echo "need exactly $required_gpus GPU ids for replicas=$REPLICAS tp=$TP; got ${#gpu_array[@]}: $GPU_IDS" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$URLS_FILE")"
: > "$URLS_FILE"

pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

for ((replica = 0; replica < REPLICAS; replica++)); do
  port=$((PORT_BASE + replica))
  offset=$((replica * TP))
  replica_gpus=("${gpu_array[@]:offset:TP}")
  replica_gpu_csv="$(IFS=,; echo "${replica_gpus[*]}")"
  url="http://${ADVERTISE_HOST}:${port}"
  printf '%s\n' "$url" >> "$URLS_FILE"
  printf 'replica=%d tp=%s gpus=%s url=%s\n' "$replica" "$TP" "$replica_gpu_csv" "$url"

  if [ "$DRY_RUN" = "1" ]; then
    continue
  fi

  # run_rollout_fp8.sh execs SGLang, so this subshell PID is the server PID.
  CUDA_VISIBLE_DEVICES="$replica_gpu_csv" \
    "$HERE/run_rollout_fp8.sh" --port "$port" --tp "$TP" \
    >"$LOG_DIR/replica-${replica}.log" 2>&1 &
  pids+=("$!")
done

printf 'wrote %d rollout URLs to %s\n' "$REPLICAS" "$URLS_FILE"
if [ "$DRY_RUN" = "1" ]; then
  trap - EXIT INT TERM
  exit 0
fi

# Keep the launcher alive as a supervisor. A failed server terminates the
# replica set rather than silently leaving a stale endpoint in ROLLOUT_URLS.
wait -n "${pids[@]}"
