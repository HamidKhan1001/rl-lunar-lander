"""Vectorized PPO training pipeline for LunarLander-v3."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from src.wrappers import FuelEfficientLandingWrapper

ENV_ID = "LunarLander-v3"


def make_env(continuous: bool = False, shaped: bool = True) -> Callable[[], gym.Env]:
    """Return a thunk that builds one LunarLander environment.

    ``Monitor`` sits *inside* the shaping wrapper, so logged episode rewards are
    always the raw environment reward and remain comparable across runs.

    Args:
        continuous: Use the continuous action space instead of Discrete(4).
        shaped: Apply :class:`FuelEfficientLandingWrapper`.
    """

    def _init() -> gym.Env:
        env: gym.Env = gym.make(ENV_ID, continuous=continuous)
        env = Monitor(env)
        if shaped:
            env = FuelEfficientLandingWrapper(env)
        return env

    return _init


def build_vec_env(
    n_envs: int, continuous: bool, shaped: bool, seed: int, subproc: bool
) -> VecEnv:
    """Create a vectorized environment (``SubprocVecEnv`` or ``DummyVecEnv``)."""
    return make_vec_env(
        make_env(continuous, shaped),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if subproc else DummyVecEnv,
    )


def train(
    total_timesteps: int = 1_000_000,
    continuous: bool = False,
    shaped: bool = True,
    n_envs: int = 8,
    seed: int = 0,
    subproc: bool = True,
    log_dir: str = "./logs",
    model_dir: str = "./models",
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
) -> Path:
    """Train PPO and persist the best checkpoint.

    Returns:
        Path to ``best_model.zip`` (best by mean *raw* evaluation reward).
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(n_envs, continuous, shaped, seed, subproc)
    # Evaluation always uses the unshaped reward so scores are comparable.
    eval_env = build_vec_env(1, continuous, False, seed + 1_000, False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_path),
        log_path=str(Path(log_dir) / "eval"),
        # eval_freq counts calls per env, so divide to get ~eval_freq total steps.
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        seed=seed,
        tensorboard_log=log_dir,
        verbose=1,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(str(model_path / "final_model"))
    finally:
        train_env.close()
        eval_env.close()
    best = model_path / "best_model.zip"
    if not best.exists():  # run shorter than one eval interval
        shutil.copyfile(model_path / "final_model.zip", best)
    return best


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--no-shaping", action="store_true", help="Disable reward shaping")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dummy-vec", action="store_true", help="Use DummyVecEnv")
    p.add_argument("--log-dir", default="./logs")
    p.add_argument("--model-dir", default="./models")
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    a = parse_args()
    best = train(
        total_timesteps=a.timesteps,
        continuous=a.continuous,
        shaped=not a.no_shaping,
        n_envs=a.n_envs,
        seed=a.seed,
        subproc=not a.dummy_vec,
        log_dir=a.log_dir,
        model_dir=a.model_dir,
    )
    print(f"Best model saved to {best}")


if __name__ == "__main__":
    main()
