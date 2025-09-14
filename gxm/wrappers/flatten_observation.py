from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.wrappers.wrapper import Wrapper


class FlattenObservation(Wrapper):
    """Wrapper that adds a rollout method to the environment."""

    def __init__(self, env: Environment):
        self.env = env

    def flatten(self, obs: Any) -> Array:
        obs_leaves = jax.tree.leaves(obs)
        obs_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in obs_leaves])
        return obs_flat

    def init(self, key: Array) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.init(key)
        timestep.obs = self.flatten(timestep.obs)
        return env_state, timestep

    def reset(
        self, key: Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        timestep.obs = self.flatten(timestep.obs)
        return env_state, timestep

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.step(key, env_state, action)
        timestep.obs = self.flatten(timestep.obs)
        return env_state, timestep
