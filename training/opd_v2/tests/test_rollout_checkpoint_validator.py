"""Synthetic full-checkpoint gate for rollout publication."""

import ast
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

CORE = Path(__file__).parents[1] / "src" / "opd_v2" / "trainer" / "core.py"


def _load_checkpoint_validator():
    """Load the production method without importing optional trainer runtimes."""
    tree = ast.parse(CORE.read_text())
    trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OPDTrainerV2"
    )
    validator = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_rollout_sink_checkpoint"
    )
    probe = ast.Module(
        body=[
            ast.ClassDef(
                name="CheckpointValidator",
                bases=[],
                keywords=[],
                body=[validator],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(probe)
    namespace = {"os": os, "torch": torch}
    exec(compile(probe, str(CORE), "exec"), namespace)
    return namespace["CheckpointValidator"]


def main():
    validator_cls = _load_checkpoint_validator()
    trainer = validator_cls()
    trainer.step = 7
    with tempfile.TemporaryDirectory() as path:
        config = {
            "architectures": ["Olmo3SinkForCausalLM"],
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "hidden_size": 8,
            "head_dim": 4,
        }
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(config, handle)
        tensors = {
            "model.embed_tokens.weight": torch.ones(8, 8, dtype=torch.bfloat16),
            "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
            "lm_head.weight": torch.ones(8, 8, dtype=torch.bfloat16),
        }
        for layer in range(2):
            prefix = f"model.layers.{layer}"
            tensors.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": torch.ones(8, 8, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.k_proj.weight": torch.ones(4, 8, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.v_proj.weight": torch.ones(4, 8, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.o_proj.weight": torch.ones(8, 8, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.q_norm.weight": torch.ones(8, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.k_norm.weight": torch.ones(4, dtype=torch.bfloat16),
                    f"{prefix}.self_attn.sinks": torch.arange(2, dtype=torch.bfloat16),
                    f"{prefix}.mlp.gate_proj.weight": torch.ones(16, 8, dtype=torch.bfloat16),
                    f"{prefix}.mlp.up_proj.weight": torch.ones(16, 8, dtype=torch.bfloat16),
                    f"{prefix}.mlp.down_proj.weight": torch.ones(8, 16, dtype=torch.bfloat16),
                    f"{prefix}.post_attention_layernorm.weight": torch.ones(
                        8, dtype=torch.bfloat16
                    ),
                    f"{prefix}.post_feedforward_layernorm.weight": torch.ones(
                        8, dtype=torch.bfloat16
                    ),
                }
            )
        checkpoint = os.path.join(path, "model.safetensors")
        save_file(tensors, checkpoint)
        trainer._validate_rollout_sink_checkpoint(path)
        manifest = json.load(open(os.path.join(path, "attention_sinks.json")))
        assert manifest["validated_checkpoint_weight_count"] == 27

        tensors.pop("model.layers.1.self_attn.q_proj.weight")
        save_file(tensors, checkpoint)
        try:
            trainer._validate_rollout_sink_checkpoint(path)
        except ValueError:
            pass
        else:
            raise AssertionError("missing projection was accepted")


if __name__ == "__main__":
    main()
    print("rollout checkpoint validator tests passed")
