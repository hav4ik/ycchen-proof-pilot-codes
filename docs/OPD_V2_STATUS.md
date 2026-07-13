# OPD v2 — project status & handoff

**Goal:** package Yi-Chia Chen's OPD v2 on-policy-distillation loop (Olmo3-32B student ← DeepSeek-V4-Flash
teacher) into a **cu128 / B200 (sm_100)** Docker image + a **Beaker launcher**, to ship to Allen AI, staying
**exactly faithful** to her code. Intended deltas only: cu128 packaging, FA2 attention-sink for B200 (no FA3),
`triton` rollout backend. Fork: `git@github.com:hav4ik/ycchen-proof-pilot-codes.git`, branch **`opd/b200-cu128`**.

**Where we are:** production image built + validated on 8×H200; Beaker config drafted; flashinfer evaluated &
rejected (triton faster on production fp8 KV). **One rebuild pending** (bake `faf4c62`) before ship. Remaining:
finish H200 agentic acceptance, then the B200 hardware pass via the Beaker 3-node smoke.

---

## 1 · IMAGE BUILD

| tag | digest | built at | contents | for |
|---|---|---|---|---|
| `chankhavu/ycchen-opd:cu128` | **`sha256:451201a8…`** | `94efb6c` | fixes #1–#10 + `JIT_CACHE_DIR` + `faf4c62` (agentic auto-scale) + seed mirror | **PRODUCTION / SHIP** — no drift (`94efb6c..HEAD` empty) |
| `chankhavu/ycchen-opd:cu128` (old) | `sha256:5e3ba5f6…` | `74c1dac` | missing `faf4c62` | superseded — do NOT ship |
| `chankhavu/ycchen-opd:cu128-flashinfer-sink` | `sha256:30994b4d…` | `321aff6` | + flashinfer sink (0.6.14) | evaluation only — NOT production |

- **Build:** `DOCKER_BUILDKIT=1 docker build -f docker/cu128/Dockerfile.ycchen-opd -t chankhavu/ycchen-opd:cu128 .`
  from repo root. Base `chankhavu/olmo3-olmocore:cu128-fa2-sink`. **Two venvs:** trainer/base = genuine cu128
  (torch 2.10, transformers 5.9, olmo3_sink_fa2); `/opt/venv/serve` = sglang **0.5.14** (cu130) run via CUDA-13
  forward-compat on <13 drivers. FA2 wheel in `docker/cu128/wheels/` (gitignored — copy into fresh worktrees).
- **Disk:** builds fill `/var/lib/docker`; if ENOSPC, `docker builder prune -af` (freed 104 GB once). ~15–30 min
  per build (cold cache); uv cache mount speeds sglang.
- **⚠️ IMAGE↔SOURCE DRIFT (the bug that bit us — now RESOLVED):** a pushed image lags the source. `5e3ba5f6` was
  built at `74c1dac`; `faf4c62` (agentic `AGENTIC_MAX_PROMPT_TOKENS` auto-scale) landed AFTER → not baked →
  scaled-smoke agentic tripped `max_prompt_tokens=100000 > max_traj=57344`. **Rule: rebuild the ship image from
  HEAD as the last step; `git log <build>..HEAD -- docker/ training/ *.sh *.py` must be EMPTY.** ✅ Done: the ship
  image `451201a8` was rebuilt from `94efb6c` and the drift check is empty. Full gate: `OPD_V2_SHIP_CHECKLIST.md`.
- **`JIT_CACHE_DIR`** (opt-in, baked in `run_{teacher,rollout,teacher_olmo3}.sh`): set it to a persistent path →
  symlinks `~/.cache/{deep_gemm,flashinfer,sglang,tvm-ffi}` there (arch+role scoped) so DeepGEMM compile cache
  survives runs/instances (kills the ~10–20 min teacher cold-warm). No-op when unset.

## 2 · BUGS / FIXES (all committed on `opd/b200-cu128`, validated on H200 — details in `OPD_V2_H200_BRINGUP_FIXES.md`)

Fixes #1–#10 (each was further down the pipeline than the last):
1. **forward-compat libcuda** — sglang 0.5.14 is CUDA-13; the 570 driver can't run it natively → bake NVIDIA's
   forward-compat `libcuda` (`/opt/cuda13-compat`), loaded by the run scripts only when driver < CUDA-13.
2. `rope_theta` KeyError (olmo2.py) — read `rope_parameters` else `config.rope_theta`, no invented default.
3. teacher health-gate timeout → `--skip-server-warmup`.
4. serve JIT needs CUDA-13 **toolkit** (nvcc/cccl) for DeepSeek topk/DeepGEMM.
5. flashinfer `curand.h` → `cuda-libraries-dev-13-0`.
6. `max_prefill_buffer_tokens` — server_args overlay dropped a stock member (first overlay-drift instance).
7. `single_round` prompt template → `distill_gen` symlink.
8. rollout "0.25 tok/s" was **teacher starvation**, NOT a forward-compat tax (misdiagnosis retracted).
9. weight-sync OOM → **`MEMFRAC=0.70`** on H200 (140GB; fp8 reload peak ~18–26GB; her 0.82 is for 180GB B200).
10. **loader.py dropped 3 flash_rl fixes** in the 0.5.14 re-derivation (CPU→CUDA guard, `dim≥2` skip-1D-norms,
    nested-proxy reload guard) — ported all 3 verbatim. **Recurring class: overlay re-derivations silently drop
    fixes → `grep -n proof-pilot` parity vs her source on every re-anchor.**

Operational gotchas (also in the bring-up doc):
- **Teacher "slow/inconsistent" prefill is usually a metric artifact + config:** the first-chunk
  `input throughput` is *idle-polluted* (tokens ÷ time-since-last-activity, incl. waiting on the rollout).
  Small prefills are launch-bound (forward-compat × many MoE launches × eager); **long trajectories** make it
  compute-bound (11264 chunks → 20–40k tok/s). `--disable-cuda-graph` on the teacher is HERS (prefill-only).
- **agentic `max_prompt_tokens > max_traj` guard:** config default 100000 is sized for her 130k prod ctx; the
  scaled smoke (57344) needs `AGENTIC_MAX_PROMPT_TOKENS ≤ max_traj` — `env_1node_smoke.sh` auto-scales to
  MAX_TRAJ/2 (= 28672) via `faf4c62`, **now baked in the ship image `451201a8`** (no manual override needed;
  the old `5e3ba5f6` required passing it by hand).
- `Scale param shape … not divisible by 3` weight-sync warning is BENIGN (her code, GQA q/k/v asymmetry).

**Loop status:** validated on H200 — rollout(fp8/triton-sink/cuda-graph/SWA/fp8-KV) → teacher `/score`
hidden-extract → reverse-KL/JSD → FSDP2 step → **weight-sync** (wv ticks) → **checkpoint** (DCP + HF) →
**resume**. **BOTH producers green:** `single_round` (loss 0.0948, wv 1–2) AND `agentic` (real producer, dsflash
seed) — agentic on `5e3ba5f6` with the `AGENTIC_MAX_PROMPT_TOKENS=28672` override: step1 loss 0.1191 → step2
0.0908, eos 100%, fail=0, rKL 0.10, weight-sync wv=1,2 (~58s). `starved_frac` high on 1-node (agentic producer
slower); inverts in prod (4 rollout nodes, WSYNC=4). **H200 functional acceptance CLOSED.**

## 3 · BEAKER CONFIG (`docker/cu128/launch/`, on `opd/b200-cu128`)

- **`run_mn_beaker.sh`** — Slurm→Beaker launcher adapter (her `run_mn_cu128.sh` is srun-based; Beaker has no
  srun). Each replica picks its role from `BEAKER_REPLICA_RANK`: **rank 0 = teacher · 1–4 = rollout · 5–7 =
  trainer (rank 5 = rdzv head + orchestrator)**. Everything else (ports, health gate, launch order, make_config,
  c10d torchrun, NCCL) is byte-for-byte hers.
- **KEY adaptation / main risk:** Beaker injects only the LEADER hostname → each replica writes its
  `BEAKER_NODE_HOSTNAME` to the shared `RUN_DIR` + barrier + read back in rank order = the slurm NODES[] list.
  The health gate is the empirical routability check. Confirm `BEAKER_NODE_HOSTNAME` peer-routability on the B200 cluster.
- **`beaker/opd_v33_b200.yaml`** — `version: v2`, `replicas: 8`, `leaderSelection`+`hostNetworking`+
  `propagateFailure`. Topology = 8 nodes × 8 B200 = **64 GPUs** (her V33: 1 teacher + 4 rollout + 3 trainer/world24).
- **Before running (all `<PLACEHOLDER>`):** cluster, budget, image, NCCL `IB_HCA`/`SOCKET_IFNAME`, `sharedMemory`,
  and Weka mounts — **the run dir must be WRITABLE** (loop coordinates through `RUN_DIR`), and `JIT_CACHE_DIR`
  should point at a fixed (not per-run) Weka path.
- **⚠️ Run the 3-node smoke first** (`replicas:3`, `TEACHER_NNODES=1 ROLLOUT_NNODES=1`, `MAX_STEPS=20`) — it
  exercises the new launcher code (gather + health gate + cross-node c10d) at 1/3 cost. Then full 64× B200.
- B200 MoE backend: `MOE_BACKEND=auto` (marlin is Hopper-only; likely `flashinfer_mxfp4`).

## 4 · FAITHFULNESS — production must equal her V33 (`env_v33_b200.sh` = her `run_agentic_mn_32b.sbatch`)

Her knobs: **LR 1e-5, constant schedule (no warmup), β=1.0 (reverse-KL), adam (0.9,0.95), MAX_STEPS 100000,
max_staleness 0, WEIGHT_SYNC_EVERY 4, TRAIN_BATCH_TRAJS 64, MEMFRAC 0.82, max_prompt_tokens 100000 (default),
MAX_TRAJ 130816, bundle caps 40k/50k, CHECKPOINT_EVERY 50.** **Smoke scales these DOWN — none may leak to prod**
(smoke: MEMFRAC 0.70, batch 4, WSYNC 1, AGENTIC_MAX_PROMPT_TOKENS MAX_TRAJ/2, MAX_TRAJ 57344, bundle 8k). See the
audit table in `OPD_V2_SHIP_CHECKLIST.md §4`.

## 5 · FLASHINFER SINK — CLOSED (`OPD_V2_FLASHINFER_SINK.md`)

Integrated (branch `opd/flashinfer-sink`), correctness-**validated** (100/100, incl. cuda-graph sink-reload →
bf16 sinks OK), but **rejected for production**: throughput flips by KV dtype — flashinfer +9% on bf16 KV but
**triton +10% on production fp8 KV**. Kept as opt-in fallback (`ATTENTION_BACKEND=flashinfer`). Not merged.

## 6 · B200 (sm_100) BRING-UP — student side VALIDATED (2026-07-13)

- **Trainer FA2 sink on sm_100 — PASS.** `python /opt/opd/opd_v2_train_smoke.py` (bare `python` = `/opt/conda`
  cu128 trainer venv): fp64-exact sink correction, **bit-exact** OPD JSD loss+grad, fwd/sink/q-k-v-grad parity,
  `torch.compile` clean, doc-isolation 0. The FA2 wheel's sm_100 kernel is correct on Blackwell.
- **Rollout on sm_100 — VALIDATED.** fp8 weights (flash_rl) + fp8-KV + triton sink → clean IMO-level Euclid proof
  via the chat endpoint on B200 (`finish_reason: stop`, correct reasoning + LaTeX). Also confirmed on H200
  native-cuda-13.2. **Teacher (DeepSeek-V4-Flash, TP4) is the remaining component.**
- **⚠️ Serve-validation lesson (cost an afternoon):** validate the serve with `POST /v1/chat/completions`
  (`temperature:0`), NOT raw `/generate`. Raw completion on a reasoning/chat model is OOD → degenerate output
  (repetition, single-token collapse, `二十一th` language-switching) that *mimics* a hardware/driver bug. It is not.
  Confirmed on the H200 (driver 595, native cuda-13.2) whose raw `/generate` had collapsed: the chat endpoint
  returns a correct proof → serve correctness is a **prompt-format** issue, not driver/arch. (A "native cuda-13
  broken / require driver-570" theory was chased and **retracted**.) The `-k fa2` test wrapper mis-picks the serve
  venv on native-cuda-13 → run the smoke directly (harness fix `8240b78`, pending rebuild). Details:
  `OPD_V2_H200_BRINGUP_FIXES.md` §Operational gotchas.

## 7 · OPEN ITEMS

1. **Final drift-clean rebuild** before Ai2 handoff — bakes `8240b78` (test-harness venv fix) + anything else B200
   surfaces; re-stamp the ship digest. (`451201a8` is valid now; its only un-baked commit is test-only.)
2. **B200 teacher** (DeepSeek-V4-Flash, TP4 → needs 4×B200) + **Beaker 3-node smoke** (multi-node launcher) → full
   64× V33. Student side (trainer FA2 + rollout) already green on sm_100.
3. (Backlog) cu129 serve rebase — cleaner than cu130+forward-compat; not required (serve works native on cuda-13).

## Doc index (all under `docs/`, on `opd/b200-cu128`)
`OPD_V2_STATUS.md` (this) · `OPD_V2_H200_BRINGUP_FIXES.md` (bugs #1–#10 + gotchas) · `OPD_V2_H200_SMOKE.md`
(finalization/acceptance checklist) · `OPD_V2_SHIP_CHECKLIST.md` (pre-ship gate) · `OPD_V2_FLASHINFER_SINK.md`
(flashinfer eval+decision) · `docker/cu128/launch/beaker/README.md` (Beaker) · `OPD_V2_ALGORITHM.md`,
`OPD_V2_CONFIG_REFERENCE.md`, `OPD_V2_PARITY_STATUS.md`.
