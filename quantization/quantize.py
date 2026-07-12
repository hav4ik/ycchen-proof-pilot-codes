#!/usr/bin/env python3
"""Quantize stage1-v2-7b (olmo3_sink) into a deployable compressed-tensors ckpt.

Run inside the isolated quant venv:
    cd quantization && .venv/bin/python quantize.py --scheme gptq-w4a16

Each scheme produces quantization/out/stage1-v2-7b-<scheme>/ with sinks merged
back and config patched for sglang's olmo2_sink serving.
"""
import argparse
import gc
import os
import time

import common

# scheme registry: name -> (compressed-tensors preset, method, needs_calibration)
SCHEMES = {
    # --- required ---
    "gptq-w4a16":  ("W4A16",       "gptq",  True),   # int4 sym g128, GPTQ
    # AWQ kept symmetric: sglang's compressed-tensors WNA16 only supports
    # symmetric int4 (asym W4A16_ASYM loads nowhere on this sglang). AWQ
    # activation-aware scale search still applies, folded into the weights.
    "awq-w4a16":   ("W4A16",       "awq",   True),   # int4 sym g128, AWQ
    "mxfp4":       ("MXFP4",       "rtn",   False),  # fp4 e8m0 g32, w4a4 microscale
    "mxfp4a16":    ("MXFP4A16",    "rtn",   False),  # fp4 e8m0 g32, weight-only
    # --- bonus ---
    "nvfp4":       ("NVFP4",       "rtn",   True),   # fp4 + fp8 group scale + global (act calib)
    "nvfp4a16":    ("NVFP4A16",    "rtn",   False),  # nvfp4 weight-only
    "w4a16-rtn":   ("W4A16",       "rtn",   False),  # int4 g128 round-to-nearest
    "fp8-dynamic": ("FP8_DYNAMIC", "rtn",   False),  # w8a8 fp8 dynamic act
}

IGNORE = ["lm_head"]  # keep the output head in bf16


def build_recipe(method: str, scheme: str, kv_cache_scheme=None):
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    if method == "gptq":
        # actorder pinned explicitly: the shipped GPTQ target was produced with static
        # activation ordering (llm-compressor 0.16.0 default); don't rely on the default.
        return GPTQModifier(targets="Linear", scheme=scheme, ignore=IGNORE,
                            actorder="static", kv_cache_scheme=kv_cache_scheme)
    if method == "awq":
        from llmcompressor.modifiers.awq import AWQModifier
        return AWQModifier(targets="Linear", scheme=scheme, ignore=IGNORE)
    if method == "rtn":
        return QuantizationModifier(targets="Linear", scheme=scheme, ignore=IGNORE,
                                    kv_cache_scheme=kv_cache_scheme)
    raise ValueError(method)


# FP8 per-tensor static KV cache scheme (sglang only supports per-tensor scalar
# k_scale/v_scale). Calibrated from the same forward as the weights -> needs calib.
KV_FP8 = {"num_bits": 8, "type": "float", "strategy": "tensor",
          "dynamic": False, "symmetric": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=sorted(SCHEMES))
    ap.add_argument("--num-calib", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    args = ap.parse_args()

    preset, method, needs_calib = SCHEMES[args.scheme]
    tag = os.environ.get("PP_QUANT_MODEL_TAG", "stage1-v2-7b")
    out_dir = f"{common.OUT_ROOT}/{tag}-{args.scheme}"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    print(f"[quantize] scheme={args.scheme} preset={preset} method={method} "
          f"calib={needs_calib} -> {out_dir}")

    import torch
    from transformers import AutoModelForCausalLM
    from llmcompressor import oneshot

    common.patch_llmcompressor()
    base = common.build_base()
    print(f"[quantize] base ready: {base}")

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda:0",
    )

    # sink-on calibration: MATCH inference. sglang serves sink-on, so we calibrate
    # sink-on by default -- calibrating in the regime you serve in is the principled
    # choice and removes a real calib/infer mismatch. (A 7B A/B at seqlen 2048 found
    # the gain marginal there, KL 0.0396->0.0368, but KL favors sink-on and the FX
    # trace handles the sink fine; the 2048 result does not justify deliberately
    # mismatching, esp. at long ctx.) Set PP_SINK_ON_CALIB=0 only for a deliberate
    # sink-off baseline. data-free RTN/FP4/FP8 run no forward, so it's a no-op there.
    sink_on = needs_calib and os.environ.get("PP_SINK_ON_CALIB", "1") == "1"
    if sink_on:
        import sink_patch
        sink_patch.patch_eager()
        sink_patch.load_sinks_into(model, common.SRC)
        model.config._attn_implementation = "eager"
        print("[quantize] sink-on calibration ENABLED (eager + per-head sinks)")

    # KV cache fp8 (calibrated per-tensor static k_scale/v_scale). Needs calib (the
    # scales are computed from the same forward), so only for gptq/awq. Long-ctx
    # calib matters here: it captures high-position post-YaRN K magnitudes.
    kv_scheme = KV_FP8 if (needs_calib and os.environ.get("PP_KV_CACHE_FP8", "0") == "1") else None
    if kv_scheme:
        print("[quantize] KV cache FP8 (per-tensor static k_scale/v_scale) ENABLED")
    recipe = build_recipe(method, preset, kv_scheme)

    kwargs = dict(model=model, recipe=recipe, output_dir=out_dir)
    detached_rotary = None
    if needs_calib:
        # GPTQ/AWQ: calibrate on real (pre-tokenized) training sequences.
        ds = common.load_calib(num_samples=args.num_calib, seqlen=args.seqlen)
        kwargs.update(dataset=ds, num_calibration_samples=len(ds),
                      max_seq_length=args.seqlen,
                      processor=common.load_fast_tokenizer())
    else:
        # data-free schemes (RTN weights + dynamic activations): no dataset,
        # weight scales come from the weights directly. The data-free pipeline
        # walks the whole module tree and llmcompressor's observe() mis-iterates
        # the per-layer RoPE `rotary_embs` ModuleDict (yields str keys -> infinite
        # recursion). It has no quantizable weights and data-free runs no forward,
        # so detach it from the tree during oneshot.
        inner = model.model
        detached_rotary = inner._modules.pop("rotary_embs", None)

    oneshot(**kwargs)

    if detached_rotary is not None:
        model.model._modules["rotary_embs"] = detached_rotary
    print(f"[quantize] oneshot done in {time.time() - t0:.0f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    common.finalize(out_dir)
    print(f"[quantize] DONE {args.scheme} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
