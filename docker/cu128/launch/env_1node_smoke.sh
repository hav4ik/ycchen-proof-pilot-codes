# OPD v2 — single-node 8-GPU (4:2:2) reduced integration smoke. Source before run_1node.sh:
#   source /opt/opd/launch/env_1node_smoke.sh && bash /opt/opd/launch/run_1node.sh
#
# Purpose: prove the full loop (rollout -> teacher hidden-extract -> JSD -> FSDP2 step -> weight sync ->
# checkpoint) end-to-end on real hardware before the 64x B200 run. Reduced scale so it fits 8 GPUs.
# REQUIRED cluster paths: STUDENT_PATH, DEEPSEEK_V4_FLASH, RUN_DIR (scratch/shared).

# core loop, no agentic pool (single_round has no min_gen_room floor -> smallest feasible context).
# Prompts come from the in-repo distill_gen/problems/problems.parquet (self-contained).
export PRODUCER=single_round
export ATTN_IMPL=olmo3_sink_fa2
export STUDENT_PATH=${STUDENT_PATH:?set to the deploy-format student dir (e.g. chankhavu/yccchen-olmo3-deploy checkout)}
export STUDENT_DEPLOY_PATH=${STUDENT_DEPLOY_PATH:-$STUDENT_PATH}   # same deploy dir works for weight-sync saves
export ROLLOUT_MODEL=${ROLLOUT_MODEL:-$STUDENT_PATH}
export DEEPSEEK_V4_FLASH=${DEEPSEEK_V4_FLASH:?set to the DeepSeek-V4-Flash checkpoint dir}

# ---- 8-GPU layout: teacher TP4 [0-3] | rollout TP2 [4-5] | trainer world2 [6-7] ----
export TEACHER_GPUS=0,1,2,3 TEACHER_TP=4
export ROLLOUT_GPUS=4,5     ROLLOUT_TP=2
export TRAINER_GPUS=6,7     TRAINER_NPROC=2

# ---- reduced scale (fits 8xH200; MICRO must be >= MAX_TRAJ_TOKENS: whole traj un-windowed) ----
export CONTEXT_LEN=24576 MAX_TRAJ_TOKENS=24576 MAX_NEW_TOKENS=20480 MICRO=24576
export KV_CACHE_DTYPE=fp8_e4m3 SWA_RATIO=0.2 ROLLOUT_MAXRUN=8 MEMFRAC=0.85
export TEACHER_CONTEXT_LEN=28672 TEACHER_MEMFRAC=0.78 TEACHER_MAXRUN=16
export TARGET_INFLIGHT=16 ROLLOUT_GEN_TIMEOUT=1800 STARVE_TIMEOUT=3600 DROP_FINISH_REASONS=length
export CHUNK_SIZE=2048 CPU_OFFLOAD=1 TRAIN_BATCH_TRAJS=8
export BETA=1.0 LR=1e-5 WEIGHT_SYNC_EVERY=4 G4_EVERY=5 LOG_EVERY=1
export CHECKPOINT_EVERY=20 CHECKPOINT_KEEP=1 HF_EXPORT=0 RESUME=0
export RUN_NAME=${RUN_NAME:-opd_1node_smoke} MAX_STEPS=${MAX_STEPS:-30}
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_PROJECT=${WANDB_PROJECT:-opd-v2-smoke}

# ---- to instead exercise the AGENTIC path (closest to the B200 production run) ----
# The agentic startup guard requires max_traj_tokens >= max(refine_cap, select_cap) + min_gen_room(48000),
# and min_gen_room is NOT env-exposed, so the floor is ~56k even with reduced bundle caps. Set:
#   export PRODUCER=agentic REFINE_BUNDLE_CAP=8000 SELECT_BUNDLE_CAP=8000
#   export CONTEXT_LEN=57344 MAX_TRAJ_TOKENS=57344 MICRO=57344 MAX_NEW_TOKENS=49152
#   export TEACHER_CONTEXT_LEN=61440
# (this is memory-heavier; if the world-2 trainer OOMs, lower TRAIN_BATCH_TRAJS or MICRO, or reclaim a
#  GPU from the teacher.) The agentic seed (public ycchen/dsflash-proof-distill-v2-test) is built at
# runtime by the orchestrator; needs network on the node.
