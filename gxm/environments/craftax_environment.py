from typing import Any

import gymnax.environments.spaces
import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree


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

    @classmethod
    def gymnax_to_gxm_space(cls, craftax_space) -> Space:
        """Convert a Gymnax space to a Gxm space."""
        if isinstance(craftax_space, gymnax.environments.spaces.Discrete):
            return Discrete(craftax_space.n)
        if isinstance(craftax_space, gymnax.environments.spaces.Box):
            return Box(
                low=craftax_space.low,
                high=craftax_space.high,
                shape=craftax_space.shape,
            )
        if isinstance(craftax_space, gymnax.environments.spaces.Dict):
            return Tree(
                {k: cls.gymnax_to_gxm_space(v) for k, v in craftax_space.spaces.items()}
            )
        if isinstance(craftax_space, gymnax.environments.spaces.Tuple):
            return Tree([cls.gymnax_to_gxm_space(s) for s in craftax_space.spaces])
        else:
            raise NotImplementedError(
                f"Gymnax space type {type(craftax_space)} not supported."
            )
