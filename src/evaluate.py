"""Deterministic evaluation and video recording for trained PPO agents."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.train import ENV_ID


def evaluate(
    model_path: str, n_episodes: int = 20, seed: int = 123
) -> tuple[float, float]:
    """Evaluate a model deterministically on the raw (unshaped) reward.

    Returns:
        ``(mean_reward, std_reward)`` over ``n_episodes``.
    """
    model = PPO.load(model_path, device="cpu")
    continuous = not hasattr(model.action_space, "n")
    env = Monitor(gym.make(ENV_ID, continuous=continuous))
    env.reset(seed=seed)
    try:
        mean, std = evaluate_policy(
            model, env, n_eval_episodes=n_episodes, deterministic=True
        )
    finally:
        env.close()
    return float(mean), float(std)


def record_video(
    model_path: str,
    video_dir: str = "./videos",
    n_episodes: int = 3,
    seed: int = 123,
) -> list[float]:
    """Roll out the policy and write MP4s to ``video_dir``.

    Returns:
        The raw return of each recorded episode.
    """
    model = PPO.load(model_path, device="cpu")
    continuous = not hasattr(model.action_space, "n")
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    env = RecordVideo(
        gym.make(ENV_ID, continuous=continuous, render_mode="rgb_array"),
        video_folder=video_dir,
        episode_trigger=lambda _: True,
        name_prefix="lunar_lander",
    )
    returns: list[float] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done, total = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
    finally:
        env.close()
    return returns


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Evaluate a trained PPO agent")
    p.add_argument("--model", default="./models/best_model.zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--video", action="store_true", help="Also record videos")
    p.add_argument("--video-dir", default="./videos")
    a = p.parse_args()

    mean, std = evaluate(a.model, a.episodes)
    print(f"Mean reward over {a.episodes} episodes: {mean:.2f} +/- {std:.2f}")
    if a.video:
        rets = record_video(a.model, a.video_dir)
        print(f"Recorded {len(rets)} episodes to {a.video_dir}: {np.round(rets, 1)}")


if __name__ == "__main__":
    main()
