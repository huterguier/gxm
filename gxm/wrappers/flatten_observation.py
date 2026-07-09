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

    def __init__(self, env: Dynamics[Any, TStep], unwrap: bool = True):
        super().__init__(env, unwrap=unwrap)

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
        env_state, step = self.env.init(key)
        return env_state, self._flatten_step(step)

    def reset(self, key: Key, env_state: DynamicsState) -> tuple[DynamicsState, TStep]:
        env_state, step = self.env.reset(key, env_state)
        return env_state, self._flatten_step(step)

    def step(
        self,
        key: Key,
        env_state: DynamicsState,
        action: PyTree,
    ) -> tuple[DynamicsState, TStep]:
        env_state, step = self.env.step(key, env_state, action)
        return env_state, self._flatten_step(step)
