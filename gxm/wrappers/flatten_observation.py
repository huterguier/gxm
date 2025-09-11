from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from gxm.core import Environment, EnvironmentState
from gxm.wrappers.wrapper import Wrapper


class FlattenObservation(Wrapper):
    """Wrapper that adds a rollout method to the environment."""

    def __init__(self, env: Environment):
        self.env = env

    def flatten(self, obs: Any) -> Array:
        obs_leaves = jax.tree.leaves(obs)
        obs_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in obs_leaves])
        return obs_flat

    def init(self, key: Array) -> EnvironmentState:
        env_state = self.env.init(key)
        env_state.obs = self.flatten(env_state.obs)
        return env_state

    def reset(self, key: Array, env_state: EnvironmentState) -> EnvironmentState:
        env_state = self.env.reset(key, env_state)
        env_state.obs = self.flatten(env_state.obs)
        return env_state

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> EnvironmentState:
        env_state = self.env.step(key, env_state, action)
        env_state.obs = self.flatten(env_state.obs)
        return env_state
