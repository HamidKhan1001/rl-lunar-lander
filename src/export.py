"""Export a trained SB3 PPO policy to ONNX and validate it."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from torch import nn

OBS_DIM = 8


class OnnxablePolicy(nn.Module):
    """Deterministic inference wrapper around an SB3 actor-critic policy.

    Only the actor path is exported (features -> shared MLP -> action head).
    Discrete policies emit ``argmax(logits)`` as int64; continuous policies emit
    the mean action clipped to the action-space bounds, matching
    ``PPO.predict(deterministic=True)``.
    """

    def __init__(self, policy: BasePolicy, continuous: bool, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.continuous = continuous
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Map a ``(batch, 8)`` observation to a deterministic action."""
        features = self.features_extractor(observation)
        latent_pi, _ = self.mlp_extractor(features)
        out = self.action_net(latent_pi)
        if self.continuous:
            return torch.max(torch.min(out, self.high), self.low)
        return torch.argmax(out, dim=1)


def export_to_onnx(
    model_path: str,
    onnx_path: str = "./models/lunar_lander_policy.onnx",
    opset: int = 17,
) -> Path:
    """Export the policy at ``model_path`` to ONNX, validating the result.

    A ``<onnx_path>.json`` metadata sidecar records the action-space type so the
    inference server can interpret the output.

    Returns:
        Path of the written ONNX file.
    """
    model = PPO.load(model_path, device="cpu")
    space = model.action_space
    continuous = not hasattr(space, "n")
    low = space.low if continuous else np.zeros(1)
    high = space.high if continuous else np.zeros(1)

    wrapper = OnnxablePolicy(model.policy, continuous, low, high).eval()
    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    out = Path(onnx_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # torch>=2.5 defaults to the dynamo exporter; pin the stable TorchScript one.
    extra = (
        {"dynamo": False}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters
        else {}
    )
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
        **extra,
    )

    onnx.checker.check_model(onnx.load(str(out)))
    # Smoke-test that the runtime can actually load and run it.
    ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
        None, {"observation": dummy.numpy()}
    )

    meta = {"continuous": continuous, "obs_dim": OBS_DIM}
    out.with_suffix(".json").write_text(json.dumps(meta))
    return out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Export PPO policy to ONNX")
    p.add_argument("--model", default="./models/best_model.zip")
    p.add_argument("--output", default="./models/lunar_lander_policy.onnx")
    a = p.parse_args()
    print(f"Exported and validated: {export_to_onnx(a.model, a.output)}")


if __name__ == "__main__":
    main()
