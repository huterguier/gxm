from typing import Any

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

from gxm.core import Environment, EnvironmentState


class CraftaxEnvironment(Environment):
    """Base class for Gymnax environments."""

    env: Any
    env_params = Any

    def __init__(self, id: str, **kwargs):
        self.env = make_craftax_env_from_name(id, auto_reset=True, **kwargs)
        self.env_params = self.env.default_params

    def init(self, key: jax.Array) -> EnvironmentState:
        obs, state = self.env.reset(key, self.env_params)
        env_state = EnvironmentState(
            state=state,
            obs=obs,
            true_obs=obs,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(False),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state

    def reset(self, key: jax.Array, env_state: EnvironmentState) -> EnvironmentState:
        del env_state
        return self.init(key)

    def step(
        self, key: jax.Array, env_state: EnvironmentState, action: jax.Array
    ) -> EnvironmentState:
        obs, state, reward, done, _ = self.env.step(
            key, env_state.state, action, self.env_params
        )
        env_state = EnvironmentState(
            state=state,
            obs=obs,
            true_obs=obs,
            reward=reward,
            terminated=done,
            truncated=done,
            info={},
        )
        return env_state

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
