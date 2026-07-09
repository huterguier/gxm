import dataclasses
from typing import Any

import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.typing import Key, PyTree
from gxm.wrappers.wrapper import EnvironmentWrapper


class IgnoreTruncation(EnvironmentWrapper[Any]):
    """
    A wrapper that treats truncation as termination.

    Truncation is folded into the terminated flag and ``true_next_obs`` is set
    equal to ``next_obs``, so downstream code sees a plain termination with no
    distinction between the two episode-ending conditions.
    """

    def __init__(self, wrapped: Environment):
        super().__init__(wrapped)

    @staticmethod
    def _remove_truncation(timestep: Timestep) -> Timestep:
        return dataclasses.replace(
            timestep,
            terminated=jnp.logical_or(timestep.terminated, timestep.truncated),
            truncated=jnp.zeros_like(timestep.truncated),
            true_next_obs=timestep.next_obs,
        )

    def init(self, key: Key) -> tuple[EnvironmentState, Timestep]:
        state, timestep = self.wrapped.init(key)
        return state, self._remove_truncation(timestep)

    def reset(
        self, key: Key, state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        state, timestep = self.wrapped.reset(key, state)
        return state, self._remove_truncation(timestep)

    def step(
        self, key: Key, state: EnvironmentState, action: PyTree
    ) -> tuple[EnvironmentState, Timestep]:
        state, timestep = self.wrapped.step(key, state, action)
        return state, self._remove_truncation(timestep)
