# OPD v2 — debugging runbook (which log, which symptom, what it means)

Distilled from bringing this loop up on H200 + B200. It is **symptom → where to look → what it means → fix**.
The deeper fix history is in [`../../../../docs/OPD_V2_H200_BRINGUP_FIXES.md`](../../../../docs/OPD_V2_H200_BRINGUP_FIXES.md);
this file is the practical "something's wrong, where do I look" guide.

---

## 0 · The log map (everything lives under `RUN_DIR`)

Get a replica's stdout with `beaker experiment logs <exp-id>` (one stream per replica), but the **real** state
is the files the launcher writes to the shared `RUN_DIR`:

Five kinds of log, one per role (plus the per-replica dispatcher). Names are exact.

| file | role | what it tells you |
|---|---|---|
| **`launch_rank<N>.log`** | dispatcher (every replica) | **start here for plumbing.** The per-replica launcher log — phase progression: host-gather → health gate → `make_config` → trainer torchrun → orchestrator. `<N>` = `BEAKER_REPLICA_RANK`. |
| **`orchestrator.log`** | orchestrator (head) | the **training heartbeat** — per-step metrics `step= loss= wv= eos= starved_frac= g4=`, weight-sync ticks, and the agentic startup guards. Lives on the head (first trainer) node. **This is where you watch the run.** |
| **`trainer_<K>.log`** | trainer (per node) | the trainer torchrun / FSDP2 log. `<K>` = trainer-local-rank (`trainer_0.log` on the head, `trainer_1.log`/`trainer_2.log` on the other trainer nodes). **CUDA OOM, c10d rendezvous, gnorm/loss** show here. |
| **`teacher_<host>_<port>.log`** | teacher `/score` | model load, DeepGEMM warmup, fp4 autotune, per-`/score` request. One per teacher replica (`TEACHERS_PER_NODE=2` → two ports). |
| **`rollout_<host>_<port>.log`** | rollout = **student** | fp8 load, triton attention-sink, `/generate`, and the **weight-sync reloads** (`update_weights_from_disk`). One per rollout replica. |

Plus the state files (not logs, but ground truth): **`config.json`** (the resolved config all 4 processes read —
`grep` it to confirm a knob took effect), **`trainer_endpoint.json`** (trainer c10d/HTTP endpoint),
**`.beaker_hosts_<id>/rank_N`** (the rank→hostname gather; a missing `rank_N` = that replica hasn't checked in),
and **`STOP`** (a sentinel — if it exists, some replica aborted and told the others to stop; find the first
failure in the launcher logs).

**Launch order (know which phase you're in):** teacher + rollout servers start → **health gate** (head waits for all
`/health` = 200) → `make_config` writes `config.json` → trainer torchrun (c10d rendezvous into one world) →
orchestrator drives steps. A hang/failure is almost always *at a phase boundary* — find the last successful phase
line in `launch_rank<N>.log`.

---

## 1 · THE #1 RULE: sanity-check the student with the **chat** endpoint, never raw `/generate`

This one cost us **three GPU rentals** and a bogus "the CUDA-13 / driver-570 build is broken" theory. Raw
`/generate` **bypasses the chat template** → the model gets an out-of-distribution prompt → it emits garbage
(repeated tokens, mid-sentence language switching). That garbage *looks exactly like a hardware/driver bug* but
is not. There is **no driver/arch dependency in serve correctness.**

```bash
# CORRECT — hits the chat template, temperature 0:
curl -s localhost:8200/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "default", "temperature": 0,
  "messages": [{"role":"user","content":"Prove that the square root of 2 is irrational."}]
}' | python -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:800])'
```
**Pass** = a coherent proof. If you instead see `n n n …` or the text flips language mid-way, you are almost
certainly hitting raw `/generate` (or feeding un-templated tokens) — switch to `/v1/chat/completions`. Only
after the chat endpoint is coherent should you suspect the stack.

---

## 2 · Symptom → diagnosis → fix

| symptom (where) | what it means | fix |
|---|---|---|
| **Teacher `/health` empty / server not "up" for ~10–16 min** (`teacher_*.log`) | The fp4 MoE **autotune** runs during init (a memory-profiling forward) **before** the port opens — it is NOT hung. Confirm: `nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv` shows the GPUs **busy**; the log shows `DeepGEMM warmup: N/N` done then silence. | Wait (~15 min cold). The baked JIT+autotune seed cuts this to ~1–2 min; if it's slow, the seed didn't copy — check for `[jit-cache] seeded …` in the log and that `JIT_CACHE_DIR` is set. |
| **Teacher crashes at startup: "Hidden size mismatch" / `swiglu_limit` / `format_is_bypassed`** (`teacher_*.log`) | Wrong MoE backend. DeepSeek-V4-Flash experts are **fp4**; every fp8 runner (`auto`→triton, `deep_gemm`, `flashinfer_trtllm`) crashes on sm_100. | Use **`flashinfer_mxfp4`** — `run_teacher.sh` **auto-detects** it on B200 now. Only override `MOE_BACKEND` to force something. |
| **Rollout output is garbage / switches language** | You're bypassing the chat template (see §1) — **not** a driver/build bug. | Validate via `/v1/chat/completions` (temp 0). |
| **Trainer `CUDA out of memory` on step 1** (`trainer_<K>.log`) | The forward+backward at `MICRO` doesn't fit. Her V33 (128k) sat at the H200 memory edge; the **Max** config (160k) is more aggressive. | Lower **`MICRO` and `MAX_TRAJ_TOKENS` together by ~8k** (her documented single knob) until it fits — no other re-analysis. |
| **Multi-node gather hangs > ~10 min** (`launch_rank*.log` stuck at "waiting for all N replicas") | Replicas can't reach each other by `BEAKER_NODE_HOSTNAME` (peer-routability — the launcher's one unverified assumption). Check `.beaker_hosts_*/` for missing `rank_N` files. | Confirm `hostNetworking: true` + `leaderSelection: true`; verify the hostnames in `.beaker_hosts_*` are routable on your cluster. |
| **`health gate TIMEOUT`** (`launch_rank<5>.log`, after 30 min) | A teacher/rollout URL/port is wrong, OR a server never became healthy (cold warmup > 30 min on a fully cold cache). | Read the named `teacher_*/rollout_*` log for that server. Seed the JIT cache (default on) so cold-start fits the 30-min gate. |
| **`agentic: max_prompt_tokens=… > max_traj_tokens=…` → SystemExit** (`orchestrator.log`) | Orchestrator startup guard: a render-pass prompt could exceed the trajectory budget. Happens when `MAX_TRAJ_TOKENS` is lowered below the `max_prompt_tokens` default (100000) without lowering it too. | Set `AGENTIC_MAX_PROMPT_TOKENS ≤ MAX_TRAJ_TOKENS` (the smoke uses `MAX_TRAJ/2`). Production/Max don't trip it (max_traj ≥ 130816 > 100000). |
| **`agentic: max_traj_tokens=… too small (need ≥ …)` → SystemExit** | The other agentic guard: `max_traj < max(bundle_cap) + min_gen_room(48000)`. | Raise `MAX_TRAJ_TOKENS` or lower the refine/select bundle caps. |
| **`starved_frac` high (e.g. 60%+)** (`orchestrator.log`) | Early = **expected** (cold pipeline; the buffer fills slowly for long-CoT). Persistent = the trainer is **rollout/teacher-starved** (not producing scored trajectories fast enough). | Early: ignore. Persistent: check the teacher is scoring (`teacher_*.log`), and consider the `ROLLOUT_NNODES=5` (1+5+2) rebalance if rollout-bound. |
| **`wv=` / weight_version not incrementing** (`orchestrator.log`) | Weight-sync isn't committing trainer→rollout. | Check the `rollout_*.log` for `update_weights_from_disk 200` + "N sink tensors reloaded". `WEIGHT_SYNC_EVERY` controls cadence (prod=4, smoke=1). |
| **`Scale param shape … not divisible by 3` warning** | **Benign** (her code) — GQA q/k/v asymmetry in the fp8 weight-sync reload. Not an error. | Ignore. |
| **Suspected bad pre-built JIT/autotune cache** (odd kernel error after a JIT_CACHE_DIR reuse) | `JIT_CACHE_DIR` is persistent, so a bad entry lingers even after flipping the seed flag. | `rm -rf "$JIT_CACHE_DIR"/sm100/` **then** set `JIT_AUTOTUNE_SEED=0` (re-tune) or `JIT_CACHE_SEED=0` (recompile all). See `AI2_HANDOFF.md §2`. |

---

## 3 · The meta-workflow (how we actually found these)

1. **Isolate the component.** Most "loop is broken" turned out to be one process. Run the teacher or rollout
   **standalone** (`run_teacher.sh --tp 4` / `run_rollout.sh --tp 4`) and validate it *alone* before blaming the
   loop. (The "0.25 tok/s rollout" was teacher-starvation, not the rollout — standalone rollout was healthy.)
2. **Separate correctness from plumbing.** Correctness → the **chat endpoint** (§1). Plumbing (gather, health,
   c10d) → the `launch_rank*.log` phase lines. Don't debug them together.
3. **"Hung" vs "slow": ask the GPU.** `nvidia-smi` — busy = it's compiling/autotuning/computing (wait); idle with
   no new log lines + a traceback = a real stall. We twice called a slow autotune "hung"; it wasn't.
4. **Trust the container, not GitHub.** A pushed image can lag the source. Verify a fix is actually *baked*:
   `docker run --rm <image> grep -n <thing> /opt/opd/...`, and check drift with
   `git log <build-commit>..HEAD -- docker/cu128/Dockerfile* docker/cu128/opd_serve 'docker/cu128/launch/*.sh' training/ '*.py'` (empty = clean).
5. **`config.json` is ground truth.** If a knob "isn't taking effect," `grep` it in `$RUN_DIR/config.json` — that's
   the resolved value all four processes actually use, after env → `make_config`.
