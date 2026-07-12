# OPD v2 — single-node 8-GPU (4:2:2) MINIMAL end-to-end shakeout. Source before run_1node.sh:
#   source /opt/opd/launch/env_1node_smoke.sh && bash /opt/opd/launch/run_1node.sh
#
# Purpose: prove the full loop works end-to-end (rollout -> teacher hidden-extract -> JSD -> FSDP2 step
# -> weight sync -> checkpoint) on real hardware before the 64x B200 run. Deliberately small: 40k max
# seq-len, small batch, ~20 steps. Every knob below is env-overridable (${VAR:-default}).
# REQUIRED cluster paths: STUDENT_PATH, DEEPSEEK_V4_FLASH, RUN_DIR (scratch/shared).

# Core loop, no agentic pool (single_round has no min_gen_room floor -> smallest feasible context).
# Prompts come from the in-repo distill_gen/problems/problems.parquet (self-contained; no HF token).
export PRODUCER="${PRODUCER:-single_round}"
export ATTN_IMPL="${ATTN_IMPL:-olmo3_sink_fa2}"
export STUDENT_PATH="${STUDENT_PATH:?set to the deploy-format student dir (e.g. chankhavu/yccchen-olmo3-deploy checkout)}"
export STUDENT_DEPLOY_PATH="${STUDENT_DEPLOY_PATH:-$STUDENT_PATH}"   # same deploy dir works for weight-sync saves
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-$STUDENT_PATH}"
export DEEPSEEK_V4_FLASH="${DEEPSEEK_V4_FLASH:?set to the DeepSeek-V4-Flash checkpoint dir}"

# ---- 8-GPU layout: teacher TP4 [0-3] | rollout TP2 [4-5] | trainer world2 [6-7] ----
export TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3}" TEACHER_TP="${TEACHER_TP:-4}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-4,5}"     ROLLOUT_TP="${ROLLOUT_TP:-2}"
export TRAINER_GPUS="${TRAINER_GPUS:-6,7}"     TRAINER_NPROC="${TRAINER_NPROC:-2}"
export MOE_BACKEND="${MOE_BACKEND:-marlin}"    # Hopper/H200: marlin OK. B200: pick a Blackwell backend.

# ---- minimal scale: 40k max seq-len, small batch (MICRO must be >= MAX_TRAJ_TOKENS: traj un-windowed) ----
export CONTEXT_LEN="${CONTEXT_LEN:-40960}" MAX_TRAJ_TOKENS="${MAX_TRAJ_TOKENS:-40960}" MICRO="${MICRO:-40960}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-36864}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}" SWA_RATIO="${SWA_RATIO:-0.2}" MEMFRAC="${MEMFRAC:-0.85}"
export TEACHER_CONTEXT_LEN="${TEACHER_CONTEXT_LEN:-45056}" TEACHER_MEMFRAC="${TEACHER_MEMFRAC:-0.78}" TEACHER_MAXRUN="${TEACHER_MAXRUN:-8}"
export ROLLOUT_N="${ROLLOUT_N:-2}"                 # samples per prompt (single_round fan-out)
export ROLLOUT_MAXRUN="${ROLLOUT_MAXRUN:-4}" TARGET_INFLIGHT="${TARGET_INFLIGHT:-8}"
export ROLLOUT_GEN_TIMEOUT="${ROLLOUT_GEN_TIMEOUT:-1800}" STARVE_TIMEOUT="${STARVE_TIMEOUT:-3600}" DROP_FINISH_REASONS="${DROP_FINISH_REASONS-length}"
export CHUNK_SIZE="${CHUNK_SIZE:-2048}" CPU_OFFLOAD="${CPU_OFFLOAD:-1}" TRAIN_BATCH_TRAJS="${TRAIN_BATCH_TRAJS:-4}"
export BETA="${BETA:-1.0}" LR="${LR:-1e-5}" WEIGHT_SYNC_EVERY="${WEIGHT_SYNC_EVERY:-4}" G4_EVERY="${G4_EVERY:-5}" LOG_EVERY="${LOG_EVERY:-1}"
# CHECKPOINT_EVERY=0 -> skip the durable DCP resume ckpt (a 32B ckpt = ~475GB of fp32 optim+master;
# not needed for a 20-step shakeout). Weight-sync still exercises the rolling weights buffer (~128GB).
# Set CHECKPOINT_EVERY=10 (+ ensure ~500GB-1TB free) to also validate the checkpoint-save path.
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}" CHECKPOINT_KEEP="${CHECKPOINT_KEEP:-1}" HF_EXPORT="${HF_EXPORT:-0}" RESUME="${RESUME:-0}"
export RUN_NAME="${RUN_NAME:-opd_1node_smoke}" MAX_STEPS="${MAX_STEPS:-20}"
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_PROJECT="${WANDB_PROJECT:-opd-v2-smoke}"

# ---- to exercise the AGENTIC path instead (closest to the B200 production run) ----
# The agentic startup guard needs max_traj_tokens >= max(refine_cap, select_cap) + min_gen_room(48000);
# min_gen_room is NOT env-exposed, so the floor is ~56k even with reduced bundle caps. Set:
#   export PRODUCER=agentic REFINE_BUNDLE_CAP=8000 SELECT_BUNDLE_CAP=8000
#   export CONTEXT_LEN=57344 MAX_TRAJ_TOKENS=57344 MICRO=57344 MAX_NEW_TOKENS=49152 TEACHER_CONTEXT_LEN=61440
# (heavier; if the world-2 trainer OOMs, lower TRAIN_BATCH_TRAJS/MICRO or set TRAINER_NPROC=3. The seed
#  is built at runtime from the public ycchen/dsflash-proof-distill-v2-test -> needs network on the node.)
