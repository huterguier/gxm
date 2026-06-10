import dataclasses

import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper


class FlattenObservation(Wrapper):
    """Wrapper that adds a rollout method to the environment."""

    def __init__(self, env: Environment, unwrap: bool = True):
        super().__init__(env, unwrap=unwrap)

    @classmethod
    def flatten(cls, obs: PyTree) -> Array:
        obs_leaves = jax.tree.leaves(obs)
        obs_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in obs_leaves])
        return obs_flat

    def _flatten_timestep(self, timestep: Timestep) -> Timestep:
        return dataclasses.replace(
            timestep,
            next_obs=self.flatten(timestep.next_obs),
        )

    def init(self, key: Key) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.init(key)
        return env_state, self._flatten_timestep(timestep)

    def reset(self, key: Key, env_state: EnvironmentState) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        return env_state, self._flatten_timestep(timestep)

    def step(
        self,
        key: Key,
        env_state: EnvironmentState,
        action: PyTree,
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.step(key, env_state, action)
        return env_state, self._flatten_timestep(timestep)
