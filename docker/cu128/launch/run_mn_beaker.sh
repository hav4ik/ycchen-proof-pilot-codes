#!/usr/bin/env bash
# OPD v2 multi-node e2e — BEAKER port of docker/cu128/launch/run_mn_cu128.sh (her V24/V33 loop).
#
# ============================ DRAFT — human review required ============================
# Not yet run. Fill the PLACEHOLDERs in beaker/opd_v33_b200.yaml + read beaker/README.md first.
# ======================================================================================
#
# run_mn_cu128.sh assumes ONE process (the slurm batch script on the head node) that fans work out
# to every node with `srun --overlap --nodelist=<n>`. Beaker has NO cross-node fan-out: it starts
# `replicas: N` IDENTICAL copies of this script, one per node, each seeing only its own 8 GPUs.
# So this port INVERTS the control flow — every replica runs THIS script and executes ONLY the role
# that its BEAKER_REPLICA_RANK maps to (rank ordering == the slurm node ordering):
#     rank 0 .. TEACHER_NNODES-1                       -> teacher node(s)
#     rank TEACHER_NNODES .. +ROLLOUT_NNODES-1         -> rollout node(s)
#     the rest                                          -> trainer node(s)
#     the FIRST trainer replica (trainer-local-rank 0) -> ALSO the rdzv head + orchestrator
#
# EVERYTHING ELSE IS COPIED VERBATIM FROM run_mn_cu128.sh — DO NOT re-tune any OPD knob here:
#   role scripts /opt/opd/opd_serve/run_{teacher,rollout}.sh, the PORT_SHIFT/port scheme, the per-role
#   env, the health gate, the launch ORDER (teacher+rollout -> health -> make_config -> trainer ->
#   orchestrator), make_config, opd_v2.trainer.service (c10d torchrun), the orchestrator invocation,
#   NCCL env, and the cleanup pkill patterns. Only Slurm->Beaker node-discovery/placement changes:
#
#   1. NODE LIST: `scontrol show hostnames $SLURM_JOB_NODELIST` (an ordered list on the head) ->
#      a shared-FS rank->hostname GATHER barrier. Beaker injects each replica's own rank + only the
#      LEADER's hostname; non-leader hostnames are NOT injected, so every replica publishes its own
#      routable hostname (BEAKER_NODE_HOSTNAME) to $RUN_DIR and all replicas read them back in rank
#      order. The reconstructed NODES[] is identical in meaning to the slurm one.     [<-- KEY RISK]
#   2. HOLDER: slurm's numeric $SLURM_JOB_ID (used for PORT_SHIFT + rdzv-id + tag) has no Beaker
#      analog that is numeric, so HOLDER = a hash of a stable per-experiment Beaker id (same on all
#      replicas). The PORT_SHIFT formula is otherwise byte-for-byte hers.
#   3. PLACEMENT: `srun --overlap --nodelist=$n <body>` -> `<body>` run locally on the one replica
#      whose rank maps to that role.
#   4. TRAINER: one cross-node `srun ... torchrun` -> each trainer replica runs its OWN torchrun;
#      they c10d-rendezvous into one world exactly as before (rdzv-endpoint = the trainer leader).
#   5. CLEANUP: srun-fanned `pkill` -> local `pkill` per replica, plus a $RUN_DIR/STOP sentinel so a
#      graceful head exit tears the other replicas down the way the slurm EXIT trap did implicitly.
#      (Crash teardown is also handled at the Beaker level by propagateFailure/propagatePreemption.)
#
# Source an env preset first, exactly as with the slurm launcher, e.g.:
#   source /opt/opd/launch/env_v33_b200.sh && bash /opt/opd/launch/run_mn_beaker.sh
set -uo pipefail
OPD_REPO="${OPD_REPO:-/opt/opd/repo}"
OPD_V2="$OPD_REPO/training/opd_v2"
ROLE_DIR="${ROLE_DIR:-/opt/opd/opd_serve}"          # baked de-apptainer'd role scripts
SRC="$OPD_V2/src"
OPD_SRC="$OPD_REPO/training/_vendor_opd"
TRAIN_PY="${OPD_TRAIN_PY:-python}"                   # image train python (base conda, transformers 5.9)
export SERVE_PY="${SERVE_PY:-/opt/venv/serve/bin/python}"   # role scripts default to this too

# ---- Beaker replica identity (replaces slurm SLURM_JOB_ID / SLURM_JOB_NODELIST / SLURM_NODEID) ----
RANK=${BEAKER_REPLICA_RANK:?must run as a Beaker replicated task (replicas: N + leaderSelection: true)}
RCOUNT=${BEAKER_REPLICA_COUNT:?BEAKER_REPLICA_COUNT missing (need replicas: N)}
# HOLDER: a numeric, deterministic, identical-across-replicas seed for PORT_SHIFT / rdzv-id / tag.
# Slurm's $SLURM_JOB_ID is numeric; no Beaker id is, so hash a stable per-experiment id to a number.
# BEAKER_EXPERIMENT_ID/BEAKER_WORKLOAD_ID are identical across all replicas of one experiment.
BK_ID="${BEAKER_EXPERIMENT_ID:-${BEAKER_WORKLOAD_ID:-${BEAKER_LEADER_REPLICA_JOB_ID:-opd_v2}}}"
HOLDER=${OPD_HOLDER:-$(( 16#$(printf '%s' "$BK_ID" | md5sum | cut -c1-7) ))}   # numeric; same on every replica
TAG=${TAG:-opd_v2_$HOLDER}
RUN_NAME=${RUN_NAME:-$TAG}
# RUN_DIR MUST be a SHARED, WRITABLE FS visible+identical on all replicas (Weka). It is the single
# source of truth: config.json, trainer_endpoint.json, teacher hidden-state spool index, weight-sync
# buffer, DCP checkpoints, the agentic pool, this gather dir, and all logs. See beaker/README.md.
RUN_DIR=${RUN_DIR:?set RUN_DIR to a SHARED, WRITABLE FS (Weka mount) visible to all replicas}
mkdir -p "$RUN_DIR"
MAIN="$RUN_DIR/launch_rank${RANK}.log"              # per-replica log (slurm wrote one head log; Beaker has N)
echo ">>> OPD v2 beaker mn launch rank=$RANK/$RCOUNT tag=$TAG run_dir=$RUN_DIR $(date)" | tee "$MAIN"

# ---- node topology: rebuild the ordered NODES[] that `scontrol show hostnames` gave on slurm ----
# Beaker gives us BEAKER_REPLICA_RANK + (leader only) BEAKER_LEADER_REPLICA_HOSTNAME. Non-leader
# hostnames are NOT injected, so each replica publishes its own routable hostname keyed by rank and
# every replica reads them back in rank order -> NODES[0]=teacher, NODES[1..]=rollout, NODES[..]=trainer.
# BEAKER_NODE_HOSTNAME is "the hostname of the node where the job is running" and, under
# hostNetworking, is the name other replicas reach it by (the leader's == BEAKER_LEADER_REPLICA_HOSTNAME).
HOSTS_DIR="$RUN_DIR/.beaker_hosts_$HOLDER"
mkdir -p "$HOSTS_DIR"
MY_HOST="${BEAKER_NODE_HOSTNAME:-$(hostname -f 2>/dev/null || hostname)}"
# canary: on the leader, BEAKER_NODE_HOSTNAME should equal the Beaker-routable leader hostname. If it
# does not, non-leader hostnames from BEAKER_NODE_HOSTNAME may also be non-routable on this cluster
# (the health gate below is the empirical check — it fails fast if a teacher/rollout URL is wrong).
if [ "$RANK" = 0 ] && [ -n "${BEAKER_LEADER_REPLICA_HOSTNAME:-}" ] && [ "$MY_HOST" != "$BEAKER_LEADER_REPLICA_HOSTNAME" ]; then
  echo "WARN: rank0 BEAKER_NODE_HOSTNAME='$MY_HOST' != BEAKER_LEADER_REPLICA_HOSTNAME='$BEAKER_LEADER_REPLICA_HOSTNAME' -> hostname routability suspect; check health gate" | tee -a "$MAIN"
fi
printf '%s\n' "$MY_HOST" > "$HOSTS_DIR/rank_${RANK}.tmp" && mv "$HOSTS_DIR/rank_${RANK}.tmp" "$HOSTS_DIR/rank_${RANK}"
echo ">>> [rank $RANK] published host '$MY_HOST'; waiting for all $RCOUNT replicas ..." | tee -a "$MAIN"
GATHER_TIMEOUT=${GATHER_TIMEOUT:-600}
t0=$(date +%s)
while :; do
  present=$(find "$HOSTS_DIR" -maxdepth 1 -name 'rank_[0-9]*' ! -name '*.tmp' 2>/dev/null | wc -l)
  [ "$present" -ge "$RCOUNT" ] && break
  [ $(( $(date +%s) - t0 )) -gt "$GATHER_TIMEOUT" ] && { echo "host-gather TIMEOUT ($present/$RCOUNT after ${GATHER_TIMEOUT}s)" | tee -a "$MAIN"; exit 1; }
  sleep 2
done
NODES=()
for r in $(seq 0 $((RCOUNT-1))); do NODES+=("$(cat "$HOSTS_DIR/rank_$r")"); done
NN=$RCOUNT

TEACHER_NNODES=${TEACHER_NNODES:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
TRAINER_NNODES=$(( NN - TEACHER_NNODES - ROLLOUT_NNODES ))
if [ "$TRAINER_NNODES" -lt 1 ]; then echo "need >= TEACHER+ROLLOUT+1 nodes (have $NN)" | tee -a "$MAIN"; exit 1; fi
TEACHER_NODES=("${NODES[@]:0:TEACHER_NNODES}")
ROLLOUT_NODES=("${NODES[@]:TEACHER_NNODES:ROLLOUT_NNODES}")
TRAINER_NODES_ARR=("${NODES[@]:TEACHER_NNODES+ROLLOUT_NNODES:TRAINER_NNODES}")
TRAINER_NODES=$(IFS=,; echo "${TRAINER_NODES_ARR[*]}")
HEAD="${TRAINER_NODES_ARR[0]}"                      # trainer rdzv head + orchestrator host (NOT the beaker leader)

TEACHER_TP=${TEACHER_TP:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHERS_PER_NODE=${TEACHERS_PER_NODE:-2}           # 2 x TP4 = 8 GPU/node
ROLLOUTS_PER_NODE=${ROLLOUTS_PER_NODE:-8}           # 8 x TP1 fp8 = 8 GPU/node
# PORT_SHIFT cap (hers): highest port (T_NCCL0 base 38600) + shift must stay <= 65535.
PORT_SHIFT=$(( (HOLDER % 800) * 32 ))
T_PORT0=$(( 8100 + PORT_SHIFT )); T_DIST0=$(( 38100 + PORT_SHIFT )); T_NCCL0=$(( 38600 + PORT_SHIFT ))
R_PORT0=$(( 8200 + PORT_SHIFT ))
TRAINER_RDZV_PORT=$(( 29500 + (PORT_SHIFT % 1000) ))
TRAINER_HTTP_PORT=${TRAINER_HTTP_PORT:-$(( 8300 + (PORT_SHIFT % 100) ))}

# Cluster-specific NCCL fabric tuning (identical to hers): set NCCL_IB_HCA / NCCL_SOCKET_IFNAME for
# your B200 fabric before launching (Beaker envVars). For ai2 IB clusters the docs suggest
# NCCL_SOCKET_IFNAME=ib and NCCL_IB_HCA=^=mlx5_bond_0 — confirm for the target B200 cluster.
NCCL_ENV="NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[ -n "${NCCL_IB_HCA:-}" ] && NCCL_ENV="NCCL_IB_HCA=$NCCL_IB_HCA $NCCL_ENV"
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && NCCL_ENV="NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME $NCCL_ENV"

# ---- role of THIS replica (rank ordering == slurm node ordering) ----
ROLE=trainer
if   [ "$RANK" -lt "$TEACHER_NNODES" ]; then ROLE=teacher
elif [ "$RANK" -lt $((TEACHER_NNODES+ROLLOUT_NNODES)) ]; then ROLE=rollout
fi
TRAINER_LOCAL_RANK=$(( RANK - TEACHER_NNODES - ROLLOUT_NNODES ))   # slurm's SLURM_NODEID analog (trainer only)
IS_HEAD=0; { [ "$ROLE" = trainer ] && [ "$TRAINER_LOCAL_RANK" = 0 ]; } && IS_HEAD=1
MY_NODE="${NODES[$RANK]}"

# Teacher/rollout endpoint URL lists (built EXACTLY as hers: http://<node>:<base+i>), computed the
# same on every replica from NODES[] + ports. The head uses them for the health gate + make_config.
TURLS=""
for ti in $(seq 0 $((TEACHER_NNODES-1))); do n="${TEACHER_NODES[$ti]}"
  for i in $(seq 0 $((TEACHERS_PER_NODE-1))); do TURLS+="http://$n:$((T_PORT0+i)),"; done; done
TURLS=${TURLS%,}
RURLS=""
for ri in $(seq 0 $((ROLLOUT_NNODES-1))); do n="${ROLLOUT_NODES[$ri]}"
  for i in $(seq 0 $((ROLLOUTS_PER_NODE-1))); do RURLS+="http://$n:$((R_PORT0+i)),"; done; done
RURLS=${RURLS%,}

echo ">>> rank=$RANK role=$ROLE node=$MY_NODE | teacher=[${TEACHER_NODES[*]}]x$TEACHERS_PER_NODE(TP$TEACHER_TP) rollout=[${ROLLOUT_NODES[*]}]x$ROLLOUTS_PER_NODE(TP$ROLLOUT_TP fp8) trainer=[$TRAINER_NODES](world=$((8*TRAINER_NNODES))) head=$HEAD" | tee -a "$MAIN"

# ---- cleanup: local only (Beaker propagateFailure handles cross-replica crash teardown) ----
cleanup() {
  echo ">>> cleanup rank=$RANK role=$ROLE tag=$TAG $(date)" | tee -a "$MAIN"
  [ "$IS_HEAD" = 1 ] && : > "$RUN_DIR/STOP" 2>/dev/null    # signal graceful teardown to the other replicas
  pkill -9 -f sglang.launch_server 2>/dev/null
  pkill -9 -f opd_v2.trainer.service 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# block until the head signals STOP, or until one of the given PIDs dies (return 1 = a child died).
serve_wait() {
  local pids=("$@")
  while :; do
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo ">>> [rank $RANK] a child ($p) died -> exit" | tee -a "$MAIN"; return 1; }; done
    [ -f "$RUN_DIR/STOP" ] && { echo ">>> [rank $RANK] STOP seen -> shutting down" | tee -a "$MAIN"; return 0; }
    sleep 15
  done
}

# =====================================================================================
# TEACHER node  (== the slurm teacher srun body, run locally on this replica)
# =====================================================================================
if [ "$ROLE" = teacher ]; then
  echo ">>> [rank $RANK] TEACHER on $MY_NODE : $TEACHERS_PER_NODE x TP$TEACHER_TP (ports $T_PORT0+)" | tee -a "$MAIN"
  PIDS=()
  for i in $(seq 0 $((TEACHERS_PER_NODE-1))); do
    gpus=$(seq -s, $((i*TEACHER_TP)) $((i*TEACHER_TP+TEACHER_TP-1)))
    port=$((T_PORT0+i)); dist=$((T_DIST0+i)); nccl=$((T_NCCL0+i))
    CUDA_VISIBLE_DEVICES=$gpus SPOOL=/dev/shm/opd-v2-tea-$port MALLOC_ARENA_MAX=4 \
      DIST_INIT_PORT=$dist SGLANG_NCCL_PORT=$nccl \
      MEMFRAC="${TEACHER_MEMFRAC:-}" MAXRUN="${TEACHER_MAXRUN:-}" \
      bash "$ROLE_DIR/run_teacher.sh" --tp "$TEACHER_TP" --port "$port" \
      > "$RUN_DIR/teacher_${MY_NODE}_$port.log" 2>&1 &
    PIDS+=($!)
  done
  serve_wait "${PIDS[@]}"; exit $?
fi

# =====================================================================================
# ROLLOUT node  (== the slurm rollout srun body, run locally on this replica)
# =====================================================================================
if [ "$ROLE" = rollout ]; then
  echo ">>> [rank $RANK] ROLLOUT on $MY_NODE : $ROLLOUTS_PER_NODE x TP$ROLLOUT_TP fp8 (ports $R_PORT0+)" | tee -a "$MAIN"
  PIDS=()
  for i in $(seq 0 $((ROLLOUTS_PER_NODE-1))); do
    gpus=$(seq -s, $((i*ROLLOUT_TP)) $((i*ROLLOUT_TP+ROLLOUT_TP-1)))
    port=$((R_PORT0+i))
    CUDA_VISIBLE_DEVICES=$gpus MALLOC_ARENA_MAX=4 \
      MODEL="${ROLLOUT_MODEL:-}" KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}" SWA_RATIO="${SWA_RATIO:-}" \
      CONTEXT_LEN="${CONTEXT_LEN:-}" MEMFRAC="${MEMFRAC:-}" MAXRUN="${ROLLOUT_MAXRUN:-}" \
      bash "$ROLE_DIR/run_rollout.sh" --port "$port" --tp "$ROLLOUT_TP" \
      > "$RUN_DIR/rollout_${MY_NODE}_$port.log" 2>&1 &
    PIDS+=($!)
  done
  serve_wait "${PIDS[@]}"; exit $?
fi

# =====================================================================================
# TRAINER node  (all trainer replicas). The head (trainer-local-rank 0) additionally runs the
# health gate + make_config BEFORE its torchrun, then the orchestrator AFTER. Non-head trainers
# wait for config.json (the make_config -> trainer order), then join the c10d rendezvous.
# =====================================================================================
run_trainer() {   # runs one 8-GPU trainer node; c10d rendezvous assigns node-rank (as in hers)
  PYTHONPATH="$SRC:$OPD_SRC" OPD_RUN_DIR="$RUN_DIR" \
  TRITON_CACHE_DIR="/tmp/triton_opdv2_$TRAINER_LOCAL_RANK" \
  env $NCCL_ENV \
    "$TRAIN_PY" -m torch.distributed.run \
      --nnodes="$TRAINER_NNODES" --nproc_per_node=8 \
      --rdzv-backend=c10d --rdzv-endpoint="$HEAD:$TRAINER_RDZV_PORT" --rdzv-id="$HOLDER" \
      -m opd_v2.trainer.service --run-dir "$RUN_DIR"
}

if [ "$IS_HEAD" = 1 ]; then
  rm -f "$RUN_DIR/trainer_endpoint.json" "$RUN_DIR/STOP"   # drop a stale endpoint/STOP from a prior round/retry

  # ---- 3) health gate (from head; teacher cold start + JIT can be ~20min) ----
  HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800}
  echo ">>> [head] health gate ($HEALTH_TIMEOUT s) over $(echo "$TURLS,$RURLS" | tr ',' ' ' | wc -w) servers ..." | tee -a "$MAIN"
  urls=$(echo "$TURLS,$RURLS" | tr ',' ' ')
  t0=$(date +%s)
  while :; do
    bad=0
    for u in $urls; do curl -s -m 3 "$u/health" >/dev/null 2>&1 || bad=$((bad+1)); done
    [ "$bad" = 0 ] && { echo "all $(echo $urls|wc -w) servers healthy" | tee -a "$MAIN"; break; }
    [ -f "$RUN_DIR/STOP" ] && { echo "STOP during health gate; abort" | tee -a "$MAIN"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt "$HEALTH_TIMEOUT" ] && { echo "health gate TIMEOUT ($bad unhealthy)" | tee -a "$MAIN"; exit 1; }
    sleep 10
  done

  # ---- 4) config.json (single source of truth) + cu128 ATTN_IMPL ----
  RUN_DIR="$RUN_DIR" RUN_NAME="$RUN_NAME" ROLLOUT_URLS="$RURLS" TEACHER_URLS="$TURLS" \
    TRAINER_HTTP_PORT="$TRAINER_HTTP_PORT" ATTN_IMPL="${ATTN_IMPL:-olmo3_sink_fa2}" \
    "$TRAIN_PY" "$OPD_V2/examples/make_config.py" 2>&1 | tee -a "$MAIN"
  [ -f "$RUN_DIR/config.json" ] || { echo "config.json not written; abort" | tee -a "$MAIN"; exit 1; }
else
  # ---- non-head trainer: wait for the head's config.json (reproduces make_config -> trainer) ----
  CONFIG_WAIT=${CONFIG_WAIT:-2400}    # must exceed HEALTH_TIMEOUT (teacher cold start) + make_config
  echo ">>> [rank $RANK] trainer waiting for $RUN_DIR/config.json (<=${CONFIG_WAIT}s) ..." | tee -a "$MAIN"
  t0=$(date +%s)
  until [ -f "$RUN_DIR/config.json" ]; do
    [ -f "$RUN_DIR/STOP" ] && { echo "STOP before config.json; abort" | tee -a "$MAIN"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt "$CONFIG_WAIT" ] && { echo "config.json wait TIMEOUT; abort" | tee -a "$MAIN"; exit 1; }
    sleep 5
  done
fi

# ---- 5) trainer: this replica's torchrun (c10d rendezvous -> one world across trainer nodes) ----
echo ">>> [rank $RANK] trainer torchrun (local-rank $TRAINER_LOCAL_RANK, world=$((8*TRAINER_NNODES))) -> rdzv $HEAD:$TRAINER_RDZV_PORT" | tee -a "$MAIN"
if [ "$IS_HEAD" = 1 ]; then
  # head: trainer in background, orchestrator in foreground (same pattern as run_1node.sh).
  run_trainer > "$RUN_DIR/trainer_$TRAINER_LOCAL_RANK.log" 2>&1 &
  TRAINER_PID=$!
  echo ">>> [head] trainer pid=$TRAINER_PID (log: trainer_$TRAINER_LOCAL_RANK.log)" | tee -a "$MAIN"

  # ---- 6) orchestrator (head node, CPU process; reads config + discovers trainer endpoint) ----
  MAX_STEPS=${MAX_STEPS:-100000}
  echo ">>> [head] launch orchestrator on $HEAD (max_steps=$MAX_STEPS)" | tee -a "$MAIN"
  PYTHONPATH="$SRC:$OPD_SRC" OPD_RUN_DIR="$RUN_DIR" CUDA_VISIBLE_DEVICES= \
    "$TRAIN_PY" -m opd_v2.orchestrator --run-dir "$RUN_DIR" --max-steps "$MAX_STEPS" \
    2>&1 | tee -a "$RUN_DIR/orchestrator.log" | tee -a "$MAIN"
  ORCH_RC=${PIPESTATUS[0]}
  echo ">>> orchestrator exited rc=$ORCH_RC $(date)" | tee -a "$MAIN"   # cleanup trap writes STOP -> tears down peers
  exit "$ORCH_RC"
else
  # non-head trainer: torchrun in the background, block until STOP (head done) or the process dies.
  run_trainer > "$RUN_DIR/trainer_$TRAINER_LOCAL_RANK.log" 2>&1 &
  serve_wait $!; exit $?
fi
