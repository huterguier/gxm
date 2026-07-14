import dataclasses
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from gxm.core import Dynamics, DynamicsState, Step, TStep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper

_TStep = TypeVar("_TStep", bound=Step)


class FlattenObservation(Wrapper[Any, TStep]):
    """Wrapper that adds a rollout method to the environment."""

    def __init__(self, wrapped: Dynamics[Any, TStep]):
        super().__init__(wrapped)

    @classmethod
    def flatten(cls, obs: PyTree) -> Array:
        obs_leaves = jax.tree.leaves(obs)
        obs_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in obs_leaves])
        return obs_flat

    def _flatten_step(self, step: _TStep) -> _TStep:
        return dataclasses.replace(
            step,
            next_obs=self.flatten(step.next_obs),
        )

    def init(self, key: Key) -> tuple[DynamicsState, TStep]:
        state, step = self.wrapped.init(key)
        return state, self._flatten_step(step)

    def reset(self, key: Key, state: DynamicsState) -> tuple[DynamicsState, TStep]:
        state, step = self.wrapped.reset(key, state)
        return state, self._flatten_step(step)

    def step(
        self,
        key: Key,
        state: DynamicsState,
        action: PyTree,
    ) -> tuple[DynamicsState, TStep]:
        state, step = self.wrapped.step(key, state, action)
        return state, self._flatten_step(step)
