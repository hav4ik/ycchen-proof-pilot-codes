# OPD v2 — her V33 production config (run_agentic_mn_32b.sbatch) + the two cu128/B200 deltas.
# Faithful to her best OPD-32B run. Source before run_mn_cu128.sh inside a >=8-node slurm allocation:
#   source /opt/opd/launch/env_v33_b200.sh && bash /opt/opd/launch/run_mn_cu128.sh
#
# REQUIRED (cluster paths): STUDENT_PATH, STUDENT_DEPLOY_PATH, ROLLOUT_MODEL, DEEPSEEK_V4_FLASH, RUN_DIR (shared FS).
# The student deploy dir works for BOTH trainer and rollout (transformers reconstructs rope_parameters
# from the legacy rope_scaling; sglang needs the legacy form) — so STUDENT_PATH may equal ROLLOUT_MODEL.

export PRODUCER=agentic
export SEED_SOURCE=${SEED_SOURCE:-ycchen/dsflash-proof-distill-v2-test}   # her OPD prompt dataset (public HF)
export SEED_HF_CONFIG=${SEED_HF_CONFIG:-per_problem}
export ATTN_IMPL=olmo3_sink_fa2          # cu128 delta #1: B200 has no FA3 -> post-correction sink on stock FA2
# (cu128 delta #2 is the rollout attention backend = triton, already the default in run_rollout.sh)

# ---- topology (her V33): 1 teacher(TP4x2) + 4 rollout(TP4x2 fp8 = 8 replicas) + 3 trainer(world 24) ----
export TEACHER_NNODES=1 TEACHER_TP=4 TEACHERS_PER_NODE=2
export ROLLOUT_NNODES=4 ROLLOUT_TP=4 ROLLOUTS_PER_NODE=2
# trainer = remaining nodes (run_mn_cu128.sh computes it; 8 total -> 3 trainer nodes -> world 24)

# ---- rollout: 140k long-context, fp8 KV, hybrid SWA ----
export CONTEXT_LEN=130816 KV_CACHE_DTYPE=fp8_e4m3 SWA_RATIO=0.2 ROLLOUT_MAXRUN=64 MEMFRAC=0.82
export MAX_TRAJ_TOKENS=130816 MAX_NEW_TOKENS=128000 ROLLOUT_GEN_TIMEOUT=6000
export TARGET_INFLIGHT=512 STARVE_TIMEOUT=7200 DROP_FINISH_REASONS=length

# ---- teacher: score window > max_traj so it never length-fails ----
export TEACHER_CONTEXT_LEN=150000 TEACHER_MEMFRAC=0.6 TEACHER_MAXRUN=16
# B200 MoE backend: run_teacher.sh defaults MOE_BACKEND=auto -> sglang picks per hardware (marlin on
# Hopper = her V33; a Blackwell backend on sm_100). DeepSeek-V4-Flash TP4 is confirmed to run on a
# GB200 node (sglang #23743). Overrides IF auto/first run misbehaves on B200:
#   export MOE_BACKEND=flashinfer_mxfp4     # the backend validated for DSv4-Flash on GB200 (#23743)
#   export MAX_PREFILL_TOKENS=8192          # only if the FlashMLA mixed decode+prefill crash appears

# ---- trainer: 32B @140k -> CPU offload, HSDP ----
export MICRO=131072 CHUNK_SIZE=2048 CPU_OFFLOAD=1 TRAIN_BATCH_TRAJS=64
export BETA=1.0 LR=1e-5 WEIGHT_SYNC_EVERY=4 G4_EVERY=5 LOG_EVERY=1
export CHECKPOINT_EVERY=50 CHECKPOINT_KEEP=2 HF_EXPORT=1 RESUME=0

export RUN_NAME=${RUN_NAME:-agentic_32b_lc140k_v33}
export MAX_STEPS=${MAX_STEPS:-100000}
export WANDB_PROJECT=${WANDB_PROJECT:-opd-v2-agentic} WANDB_MODE=${WANDB_MODE:-online}
