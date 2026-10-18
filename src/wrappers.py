"""Custom Gymnasium wrappers for reward shaping on LunarLander-v3."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


class FuelEfficientLandingWrapper(gym.RewardWrapper):
    """Reward shaping that discourages wasteful thrusting and excessive tilt.

    The default LunarLander reward already rewards proximity, low velocity and
    leg contact while charging for fuel. This wrapper adds two extra terms:

    * **Thruster penalty** -- proportional to main-engine and side-engine usage.
      Works for both the Discrete(4) action space (0=noop, 1=left, 2=main,
      3=right) and the Box(2) continuous space (main throttle, lateral throttle).
    * **Tilt penalty** -- proportional to the amount the lander's angle exceeds
      a tolerance. The tolerance widens with horizontal offset ``|x|`` because a
      lean is needed to translate sideways, but a lander hovering above the pad
      should stay upright.

    Args:
        env: The LunarLander environment to wrap.
        main_thruster_penalty: Cost per step at full main-engine throttle.
        side_thruster_penalty: Cost per step at full side-engine throttle.
        tilt_penalty: Cost per radian of excess tilt per step.
        tilt_tolerance: Base angle (radians) tolerated at ``x == 0``.
        tilt_offset_slack: Extra tolerated angle (radians) per unit of ``|x|``.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        main_thruster_penalty: float = 0.10,
        side_thruster_penalty: float = 0.03,
        tilt_penalty: float = 0.50,
        tilt_tolerance: float = 0.10,
        tilt_offset_slack: float = 0.50,
    ) -> None:
        super().__init__(env)
        self.main_thruster_penalty = main_thruster_penalty
        self.side_thruster_penalty = side_thruster_penalty
        self.tilt_penalty = tilt_penalty
        self.tilt_tolerance = tilt_tolerance
        self.tilt_offset_slack = tilt_offset_slack
        self._continuous = isinstance(env.action_space, gym.spaces.Box)
        self._last_action: Any = None
        self._last_obs: np.ndarray | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and cache the initial observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_action = None
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the env, caching the action so :meth:`reward` can use it.

        ``RewardWrapper.reward`` only receives the scalar reward, so the action
        and resulting observation are stashed here before delegating.
        """
        self._last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        shaped = self.reward(reward)
        info["raw_reward"] = float(reward)
        info["shaping"] = float(shaped) - float(reward)
        return obs, shaped, terminated, truncated, info

    def _thruster_usage(self) -> tuple[float, float]:
        """Return (main, side) engine usage in ``[0, 1]`` for the last action."""
        action = self._last_action
        if action is None:
            return 0.0, 0.0
        if self._continuous:
            a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
            # Box2D: main fires only for a[0] > 0 (throttle 50-100%),
            # side fires only for |a[1]| > 0.5 (throttle 50-100%).
            main = float(max(a[0], 0.0))
            side = float(abs(a[1])) if abs(a[1]) > 0.5 else 0.0
            return main, side
        act = int(np.asarray(action).item())
        return (1.0 if act == 2 else 0.0), (1.0 if act in (1, 3) else 0.0)

    def reward(self, reward: SupportsFloat) -> float:
        """Return the shaped reward for the most recent transition."""
        shaped = float(reward)
        main, side = self._thruster_usage()
        shaped -= self.main_thruster_penalty * main
        shaped -= self.side_thruster_penalty * side

        if self._last_obs is not None:
            x, angle = float(self._last_obs[0]), float(self._last_obs[4])
            allowed = self.tilt_tolerance + self.tilt_offset_slack * abs(x)
            shaped -= self.tilt_penalty * max(0.0, abs(angle) - allowed)
        return shaped
