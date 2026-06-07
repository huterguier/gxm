import dataclasses

import jax.numpy as jnp
from jax import Array

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.wrappers.wrapper import Wrapper


class IgnoreTruncation(Wrapper):
    """
    A wrapper that treats truncation as termination.

    Truncation is folded into the terminated flag and ``true_next_obs`` is set
    equal to ``next_obs``, so downstream code sees a plain termination with no
    distinction between the two episode-ending conditions.
    """

    def __init__(self, env: Environment):
        super().__init__(env)

    @staticmethod
    def _remove_truncation(timestep: Timestep) -> Timestep:
        return dataclasses.replace(
            timestep,
            terminated=jnp.logical_or(timestep.terminated, timestep.truncated),
            truncated=jnp.zeros_like(timestep.truncated),
            true_next_obs=timestep.next_obs,
        )

    def init(self, key: Array) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.init(key)
        return env_state, self._remove_truncation(timestep)

    def reset(
        self, key: Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        return env_state, self._remove_truncation(timestep)

    def step(
        self, key: Array, env_state: EnvironmentState, action: Array
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.step(key, env_state, action)
        return env_state, self._remove_truncation(timestep)
