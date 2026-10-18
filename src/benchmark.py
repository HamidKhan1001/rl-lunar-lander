"""Measure episode quality, fuel usage and PyTorch-vs-ONNX latency."""

from __future__ import annotations

import argparse
import json
import time

import gymnasium as gym
import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO

from src.train import ENV_ID


def rollout_stats(model_path: str, episodes: int, seed: int = 1000) -> dict[str, float]:
    """Run deterministic episodes on the raw env and summarize them.

    Reports mean/std return, the fraction of episodes scoring >= 200 (the
    environment's "solved" threshold) and main-engine firings per episode.
    """
    model = PPO.load(model_path, device="cpu")
    env = gym.make(ENV_ID)
    returns, main_fires, side_fires = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total, main, side = False, 0.0, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            a = int(action)
            main += a == 2
            side += a in (1, 3)
            obs, r, term, trunc, _ = env.step(a)
            total += float(r)
            done = term or trunc
        returns.append(total)
        main_fires.append(main)
        side_fires.append(side)
    env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(np.array(returns) >= 200)),
        "main_fires_per_ep": float(np.mean(main_fires)),
        "side_fires_per_ep": float(np.mean(side_fires)),
    }


def latency_ms(model_path: str, onnx_path: str, n: int = 2000) -> dict[str, float]:
    """Median single-state latency (ms) for SB3 ``predict`` and ONNX Runtime."""
    model = PPO.load(model_path, device="cpu")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    obs = np.random.default_rng(0).uniform(-1, 1, (1, 8)).astype(np.float32)

    def timeit(fn) -> float:
        for _ in range(100):  # warmup
            fn()
        times = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1000)
        return float(np.median(times))

    pt = timeit(lambda: model.predict(obs, deterministic=True))
    ox = timeit(lambda: session.run(None, {"observation": obs}))
    return {"sb3_ms": pt, "onnx_ms": ox, "speedup": pt / ox}


def main() -> None:
    """CLI entry point; prints one JSON document."""
    p = argparse.ArgumentParser(description="Benchmark a trained model")
    p.add_argument("--model", default="./models/best_model.zip")
    p.add_argument("--onnx", default="./models/lunar_lander_policy.onnx")
    p.add_argument("--episodes", type=int, default=100)
    a = p.parse_args()
    out = {"rollout": rollout_stats(a.model, a.episodes), "latency": latency_ms(a.model, a.onnx)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
