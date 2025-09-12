from typing import Any

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

from gxm.core import Environment, EnvironmentState, Timestep


class CraftaxEnvironment(Environment):
    """Base class for Craftax environments."""

    env: Any
    env_params = Any

    def __init__(self, id: str, **kwargs):
        self.env = make_craftax_env_from_name(id, auto_reset=True, **kwargs)
        self.env_params = self.env.default_params

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        obs, craftax_state = self.env.reset(key, self.env_params)
        env_state = craftax_state
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(False),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(
        self, key: jax.Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: jax.Array, env_state: EnvironmentState, action: jax.Array
    ) -> EnvironmentState:
        craftax_state = env_state
        obs, craftax_state, reward, done, _ = self.env.step(
            key, craftax_state, action, self.env_params
        )
        env_state = craftax_state
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=reward,
            terminated=done,
            truncated=done,
            info={},
        )
        return env_state, timestep

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
