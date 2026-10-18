"""Dev helper: build a throwaway (barely trained) model so the API runs without a real training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.export import export_to_onnx
from src.train import build_vec_env


def build_dummy(out_dir: str = "./models_dev", continuous: bool = False) -> Path:
    """Train ~2k steps, export to ONNX, and return the ONNX path.

    The policy is essentially random. It exists only to exercise the export and
    serving code in development; never report its numbers.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = build_vec_env(1, continuous, True, seed=0, subproc=False)
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, seed=0, device="cpu")
    model.learn(total_timesteps=2048)
    env.close()
    model.save(str(out / "dummy_model"))
    return export_to_onnx(str(out / "dummy_model.zip"), str(out / "lunar_lander_policy.onnx"))


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Build a dummy model for local dev")
    p.add_argument("--out-dir", default="./models_dev")
    p.add_argument("--continuous", action="store_true")
    a = p.parse_args()
    path = build_dummy(a.out_dir, a.continuous)
    print(f"Dummy ONNX model at {path}\nServe it: MODEL_PATH={path} uvicorn src.server:app")


if __name__ == "__main__":
    main()
