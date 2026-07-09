import dataclasses
from typing import Any

import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import EnvironmentWrapper


class ClipReward(EnvironmentWrapper[Any]):
    """
    Wrapper that clips the reward to a specified range.
    """

    env: Environment

    def __init__(
        self, env: Environment, unwrap: bool = True, min: float = -1.0, max: float = 1.0
    ):
        """
        Args:
            env: The environment to wrap.
            min: Minimum reward value.
            max: Maximum reward value.
        """
        super().__init__(env, unwrap=unwrap)
        self.min = min
        self.max = max

    def clip(self, reward: Array) -> Array:
        return jnp.clip(reward, self.min, self.max)

    def _clip_timestep(self, timestep: Timestep) -> Timestep:
        clipped = self.clip(timestep.reward)
        return dataclasses.replace(
            timestep,
            reward=clipped,
            info=timestep.info | {"true_reward": clipped},
        )

    def init(self, key: Key) -> tuple[EnvironmentState, Timestep]:
        state, timestep = self.env.init(key)
        return state, self._clip_timestep(timestep)

    def reset(
        self, key: Key, state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        state, timestep = self.env.reset(key, state)
        return state, self._clip_timestep(timestep)

    def step(
        self,
        key: Key,
        state: EnvironmentState,
        action: PyTree,
    ) -> tuple[EnvironmentState, Timestep]:
        state, timestep = self.env.step(key, state, action)
        return state, self._clip_timestep(timestep)
