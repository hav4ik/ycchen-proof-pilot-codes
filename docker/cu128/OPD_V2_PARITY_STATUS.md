# OPD v2 cu128 container — status, patches, and cu13↔0.5.14 parity

Container: **`ycchen-opd`** (cu128, B200-target). Two isolated venvs, all 4 OPD processes.
Student = `chankhavu/yccchen-olmo3-deploy`; teacher = DeepSeek-V4-Flash.

## 1. What we built
| process | venv | stack |
|---|---|---|
| trainer + orchestrator | base `/app/.venv` | torch 2.11+cu128 · transformers **5.9.0** (her pin) · `olmo3_sink_fa2` · liger |
| rollout (student) | `/opt/venv/serve` | **sglang 0.5.14** + her patched `olmo2.py`/`loader.py`/`load_config.py`/`server_args.py` · triton sink |
| teacher (DeepSeek-V4) | `/opt/venv/serve` | sglang 0.5.14 + re-anchored hidden-extract (`_patch_sglang_514.py`, 5 patches) |

## 2. Trainer sink — `olmo3_sink_fa2` (new backend, validated)
FA3 is Hopper-only and stock transformers `flash_attention_2` **silently drops the sink** (only FA3 `s_aux` / FA4 `learnable_sink` are honored). So we run the sink as a **post-correction re-normalization on stock FA2** (`o_sink = out·exp(lse−logaddexp(lse,sink))`), wrapped in a `torch.library.custom_op` (torch.compile-safe), in-place rescale (zero extra memory).
Validated in-container (3090s): fp64-exact vs eager (≤1.8e-15 fwd+grads); bit-exact OPD JSD loss; sink trains through the loss; packed varlen zero cross-doc leak; `torch.compile(fullgraph=True)` no graph break; **0 MB fwd+bwd memory overhead at 128K on the real 40:8/d128 deploy shape**. Cross-framework: Yi-Chia ≡ Prime-RL ≡ OLMo-core logits **bit-identical**.

## 3. Adversarial sub-agent findings (4 agents, each tried to REFUTE)
- **Agent 1 (output-processor restructure): 🔴 BUG → FIXED.** `is_emit_rank = getattr(self,"attn_tp_rank",0)==0` — in 0.5.12 `self` was the Scheduler (has `attn_tp_rank`); in 0.5.14 the code moved to a **frozen-slots dataclass** with no such attr, so it was **always True → every TP rank wrote the same spool file** (race/corruption). Fixed: compute rank via global `get_attention_tp_rank()` inside `_pp_store_hidden` (lazy import).
- **Agent 4 (end-to-end path): 🔴 BUG → FIXED (in source, rebuild pending).** 0.5.14 moved the hidden emit into `scheduler_components/output_streamer.py:481` `hs = hs[:req.finished_len]`. `/score` uses `max_new_tokens=1` → `finished_len=1`, and our patch appends **one entry per prefill CHUNK**, so any prompt **> `--chunked-prefill-size` (11264)** had its tail hidden **silently truncated to chunk 0**. Fixed: added a **5th patch** (`patch_output_streamer`) skipping the truncation in spool mode.
- **Agent 2 (deepseek hidden gate): ✅ EQUIVALENT.** Gate lands correctly; `hidden_states_before_norm=None` → logits_processor returns the post-norm hidden (verified `logits_processor.py:628-631`). Surfaced the fp8 finding below.
- **Agent 3 (scheduler / `/score` / fp8 guard): ✅ EQUIVALENT.** Widened scheduler condition is harmless (extra field never read, can't crash). `/score` distinct from native `/v1/score`, handler mirrors native generate. fp8 guard not inverted.
- **Agent 4 Part B (rollout sink parity): ✅ CLEAN.** `olmo2.py`, `radix_attention.py` **byte-identical** cu13↔cu128 (md5 match); triton sink threading identical; container's extra split-KV path is AMD-only + sink-gated (inert).

Helper-level parity (unit-tested in-container): `_dequant_fp8_wo_a_streaming` == bulk dequant + **byte-identical to hers**; `_pp_store_hidden` correct in all modes + **byte-identical to hers**. Re-anchoring changed only WHERE patches attach, not the logic.

## 4. Yi-Chia's cu13 version vs our cu128 0.5.14 version — the differences
| aspect | her (cu13, sglang 0.5.14.dev / SIF 0.5.12.post1) | ours (cu128, sglang 0.5.14 release) | why |
|---|---|---|---|
| CUDA | 13.0 (won't run on cu128 cluster) | 12.8 | AI2 ceiling; cu13 > cu128 |
| trainer sink | patched-FA3 in-kernel (Hopper) | `olmo3_sink_fa2` post-correction on stock FA2 | B200 has no FA3 |
| rollout sink model | `olmo2.py` (her patch) | **same file, byte-identical** | copied hers |
| rollout attn backend | **FA3** (H200 default; her `run_rollout_fp8.sh` does NOT pin `--attention-backend`, and its comments optimize KV dtype to *keep FA3*: `fp8_e4m3` keeps FA3, `fp8_e5m2` would force triton). sglang's FA3 backend honors the sink. | **triton** (B200 has no FA3; triton is the Blackwell-capable backend that still honors the sink — flashinfer drops it) | **real, forced substitution** — NOT "unchanged". Correctness rests on triton applying the sink identically to FA3 (verified: 0.5.14 native radix `sinks` + triton sink lines). |
| teacher patches | `_patch_sglang.py` on 0.5.12 layout | `_patch_sglang_514.py` re-anchored (5 patches) | 0.5.14 moved files |
| teacher emit-rank | `self.attn_tp_rank` (Scheduler) | `get_attention_tp_rank()` | 0.5.14 dataclass move |
| teacher chunk truncation | owned by her 0.5.12 output path | +5th patch to `output_streamer.py` | 0.5.14 split the emit |
| `wo_a` precision | dequant-to-bf16 (her default) | **`SGLANG_OPT_FP8_WO_A_GEMM=0`** → dequant-to-bf16 | fidelity + parity; fp8 GEMM would bias the teacher hidden (OPD policy bias) |
| rollout SWA KV pool | `flash_rl/patches/model_config.py` (cu13-patched file, `--bind`) | her idempotent `apply_swa_patch.py` run against **stock 0.5.14** `model_config.py` at build (anchors verified present; `is_hybrid_swa_model(Olmo3Sink)=True`) | avoids cu13→cu128 version-skew of copying her 73KB file; adds Olmo3Sink to `hybrid_swa_archs` so `--swa-full-tokens-ratio` isn't silently inert → 140k rollout fits |

## 5. Git — done
Fork `git@github.com:hav4ik/ycchen-proof-pilot-codes.git`, branch **`opd/b200-cu128`** (pushed), off base `74faacb`:
- `3a4fb96` — snapshot of pre-existing local files I did NOT author (`clients.py`, `orchestrator.py`, `core.py`, `build_l4.py`), isolated so they don't mix with our work.
- `acbb0db` — **our work**: `olmo3_sink/olmo3_sink_fa2.py` (new), `training/teacher_extract/_patch_sglang_514.py` (new), `olmo3_sink/register.py` + `training/opd_v2/examples/make_config.py` (edited).

### Her `main` moved forward (`74faacb..bc03a2c`, fetched) — purely ADDITIVE, no conflict
Our parity basis is **untouched** by her update (verified: `_patch_sglang.py`, `_patched/`, `teacher_patch/`, `olmo3_sink/`, trainer `core.py`/`orchestrator.py` — all unchanged since our base). Her new content:
- `training/opd_v2/examples/run_mn.sh` + `run_agentic_mn_32b.sbatch` — **production multi-node launcher** (slurm+apptainer): teacher(TP4×2/node) · rollout(fp8 TP1×8/node) · trainer(torchrun world=8×N c10d) · orchestrator(head CPU). This is the **reference for our deferred Beaker wiring** — it calls the exact per-role scripts our container de-apptainer'd (`run_teacher_fs.sh`→`run_teacher.sh`, `run_rollout_fp8.sh`, `make_config.py`, `opd_v2.trainer.service`, `opd_v2.orchestrator`).
- `training/dflash/canonical_fa3_train{,_prod}.py` — **NOT the OPD trainer.** This is the **DFlash speculative-decoding *draft* trainer** (an 8-layer OLMo3-sink draft model for block-diffusion drafting of the deployed target). Separate pipeline stage (deploy-time spec-decode), separate model. The OPD *student* trainer is `opd_v2.trainer`, which her update leaves unchanged.
- `deploy/prepare_yarn_deploy.py` — yarn-rope deploy prep (relevant to producing `chankhavu/yccchen-olmo3-deploy`).
- README/eval-results/quantization/kaggle — docs + numbers. The opd_v2 README update **confirms our re-anchoring**: her canonical generator patches `scheduler_output_processor_mixin.py` (the 0.5.12 name); 0.5.14 split it into `batch_result_processor.py` + `output_streamer.py` — exactly why we needed the 5th patch.

## 6. Rebuild — done (image `ycchen-opd:latest` = sha256 `97e56de7…`, 25.6 GB)
Rebuilt 2026-07-11; both patches confirmed **persisted in the final tagged image**:
- `is_emit_rank` + `output_streamer` (5th) teacher fixes → `output_streamer` spool-gate present = True.
- **SWA hybrid KV-pool** patch (new) → `is_hybrid_swa_model(Olmo3Sink) = True` (stock returns False). Build-time verify line: `serve OK: Olmo3Sink model + radix sinks + triton extend/decode sink-lines 6 + SWA hybrid-pool patched`.
- Rollout sink verified correct in **both** phases: triton `forward_extend` (prefill) *and* `forward_decode` (decode) thread `sinks=` — the sink is applied to every decoded token, not just prefill. (This is why flashinfer is unsafe: its decode path drops the sink.)

## 7. Pending
- **B200 remaining**: DeepSeek-V4-Flash MoE backend (`marlin` → Blackwell/sm_100 cutlass/triton/flashinfer) — a serving-flag benchmark, not a patch. Actual B200 run (kernels exercise on sm_100).
- **Multi-node/Beaker launch wiring**: adapt her `training/opd_v2/examples/run_mn.sh` (production topology 1 teacher + 4 rollout + 3 trainer / world 24; env overrides in `run_agentic_mn_32b.sbatch`) — swap `apptainer exec $SIF`→our two venvs, slurm→Beaker. Prereqs her sbatch documents: build the deploy-format variant (`deploy/make_olmo3sink_deploy.py`) for rollout+weight-sync, and pre-seed the pool (`opd_v2.agentic.seed`).
- **opd-image artifacts** (Dockerfile.ycchen-opd + this doc) are local, uncommitted (the `ycchen-proof-pilot-codes` fork branch `opd/b200-cu128` is pushed).
