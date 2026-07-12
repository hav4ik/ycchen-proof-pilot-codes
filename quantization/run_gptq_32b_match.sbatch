#!/bin/bash
# 32B GPTQ-w4a16 that MATCHES inference end-to-end:
#   sink-ON calibration (gpt-oss sink in the eager calib forward, = what sglang serves)
#   + long-context seqlen 10240 (>YaRN 8192 & >SWA 4096 -> activates the long-range
#     regime + high-position post-YaRN K that 2048 never sees)
#   + factor-32 YaRN rope (from the deploy config, verbatim)
#   + FP8 per-tensor static k_scale/v_scale (calibrated in the SAME forward).
# i.e. calib distribution == serving distribution on every axis we control.
#
# Submit from the repo root (or set PP_ROOT). Needs 2 GPUs.
#SBATCH --job-name=q-gptq-32b-match
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=4:00:00
#SBATCH --output=quantization/logs_gptq_32b_match_%j.out

set -uo pipefail
REPO="${PP_ROOT:-$PWD}"
cd "$REPO/quantization"
PY=.venv/bin/python   # isolated quantization venv (see quantization/README.md)

export PP_QUANT_SRC="${PP_QUANT_SRC:-$REPO/outputs/agentic_32b_lc140k_v33-s200-deploy}"
export PP_QUANT_BASE="${PP_QUANT_BASE:-$REPO/quantization/base-32b-s200}"
export PP_QUANT_MODEL_TAG="${PP_QUANT_MODEL_TAG:-agentic-32b-s200-lc10k-kv}"
export PP_SINK_ON_CALIB=1     # MATCH inference: sink-on calibration
export PP_KV_CACHE_FP8=1       # calibrate per-tensor static fp8 k_scale/v_scale

$PY -c "import torch;print('[env] cuda',torch.cuda.is_available(),'ndev',torch.cuda.device_count())"

echo "===== STEP 1: sink-ON long-ctx GPTQ + KV fp8 scale (n=64, seqlen 10240) ====="
CUDA_VISIBLE_DEVICES=0 $PY quantize.py --scheme gptq-w4a16 --num-calib 64 --seqlen 10240
echo "===== STEP 1 OK -> out/${PP_QUANT_MODEL_TAG}-gptq-w4a16 ====="

echo "===== STEP 2: verify k_scale/v_scale + sinks + kv_cache_scheme ====="
$PY - <<'EOF'
import json, glob
from safetensors import safe_open
d = "out/agentic-32b-s200-lc10k-kv-gptq-w4a16"
ks = []
for f in glob.glob(f"{d}/*.safetensors"):
    with safe_open(f, framework="pt") as h: ks += list(h.keys())
nk = sum(k.endswith('.k_scale') for k in ks)
nv = sum(k.endswith('.v_scale') for k in ks)
ns = sum(k.endswith('.self_attn.sinks') for k in ks)
print(f"k_scale={nk}  v_scale={nv}  sinks={ns}  (expect 64/64/64)")
c = json.load(open(f"{d}/config.json"))
print("kv_cache_scheme:", c['quantization_config'].get('kv_cache_scheme'))
for f in glob.glob(f"{d}/*.safetensors"):
    with safe_open(f, framework="pt") as h:
        for k in sorted(h.keys()):
            if k.endswith('.k_scale'):
                print("sample", k, "=", float(h.get_tensor(k).reshape(-1)[0])); raise SystemExit
EOF

echo "===== STEP 3: sink-ON ablation (served regime) @8192 -- long+sink-on vs old short(2048) cal ====="
$PY ablation.py --tag agentic-32b-s200-lc10k-kv --schemes gptq-w4a16 --sink-on --seqlen 8192 --n 16 \
    --out phase3_lc10k_sinkon_8192.json
$PY ablation.py --tag agentic-32b-s200          --schemes gptq-w4a16 --sink-on --seqlen 8192 --n 16 \
    --out phase3_short_sinkon_8192.json
$PY - <<'EOF'
import json
for nm, f in [("long-ctx + sink-on cal", "phase3_lc10k_sinkon_8192"),
              ("old short(2048) cal", "phase3_short_sinkon_8192")]:
    d = json.load(open(f + ".json")); g = d["gptq-w4a16"]; r = d["bf16-ref"]
    print(f"{nm:24s} @8192 sink-on: KL {g['kl']:.4f}  agree {g['top1_agree']:.4f}  "
          f"ppl +{(g['ppl']/r['ppl']-1)*100:.2f}%")
EOF
echo "===== DONE ====="
