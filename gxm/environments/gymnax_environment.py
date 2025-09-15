from typing import Any

import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree


class GymnaxEnvironment(Environment):
    """Base class for Gymnax environments."""

    env: gymnax.environments.environment.Environment
    env_params: Any

    def __init__(self, id: str, **kwargs):
        self.env, self.env_params = gymnax.make(id, **kwargs)
        self.action_space = self.gymnax_to_gxm_space(
            self.env.action_space(self.env_params)
        )

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        obs, gxm_state = self.env.reset(key, self.env_params)
        env_state = gxm_state
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
    ) -> tuple[EnvironmentState, Timestep]:
        gymnax_state = env_state
        obs, gymnax_state, reward, done, _ = self.env.step(
            key, gymnax_state, action, self.env_params
        )
        env_state = gymnax_state
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=reward,
            terminated=done,
            truncated=done,
            info={},
        )
        return env_state, timestep

    @classmethod
    def gymnax_to_gxm_space(cls, gymnax_space) -> Space:
        """Convert a Gymnax space to a Gxm space."""
        if isinstance(gymnax_space, gymnax.environments.spaces.Discrete):
            return Discrete(gymnax_space.n)
        if isinstance(gymnax_space, gymnax.environments.spaces.Box):
            return Box(
                low=gymnax_space.low,
                high=gymnax_space.high,
                shape=gymnax_space.shape,
            )
        if isinstance(gymnax_space, gymnax.environments.spaces.Dict):
            return Tree(
                {k: cls.gymnax_to_gxm_space(v) for k, v in gymnax_space.spaces.items()}
            )
        if isinstance(gymnax_space, gymnax.environments.spaces.Tuple):
            return Tree([cls.gymnax_to_gxm_space(s) for s in gymnax_space.spaces])
        else:
            raise NotImplementedError(
                f"Gymnax space type {type(gymnax_space)} not supported."
            )
