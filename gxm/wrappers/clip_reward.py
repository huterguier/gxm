import dataclasses

import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper


class ClipReward(Wrapper):
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
        env_state, timestep = self.env.init(key)
        return env_state, self._clip_timestep(timestep)

    def reset(
        self, key: Key, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        return env_state, self._clip_timestep(timestep)

    def step(
        self,
        key: Key,
        env_state: EnvironmentState,
        action: PyTree,
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.step(key, env_state, action)
        return env_state, self._clip_timestep(timestep)
