from dataclasses import dataclass
from typing import Any

import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key


@jax.tree_util.register_dataclass
@dataclass
class GymnaxEnvironmentState(EnvironmentState):
    """State for Gymnax environments."""

    gymnax_state: gymnax.EnvState
    """ The Gymnax environment state. """


class GymnaxEnvironment(Environment):
    """Base class for Gymnax environments."""

    gymnax_id: str
    """ The Gymnax environment ID. """
    env: gymnax.environments.environment.Environment
    """ The Gymnax environment. """
    env_params: Any
    """ The parameters for the Gymnax environment. """

    def __init__(self, id: str, **kwargs):
        self.id = id
        self.gymnax_id = id.split("/", 1)[1]
        self.env, self.env_params = gymnax.make(self.gymnax_id, **kwargs)
        self.action_space = self.gymnax_to_gxm_space(
            self.env.action_space(self.env_params)
        )
        self.observation_space = self.gymnax_to_gxm_space(
            self.env.observation_space(self.env_params)
        )

    def init(self, key: Key) -> tuple[GymnaxEnvironmentState, Timestep]:
        obs, gxm_state = self.env.reset(key, self.env_params)
        env_state = GymnaxEnvironmentState(gymnax_state=gxm_state)
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(
        self, key: Key, env_state: GymnaxEnvironmentState
    ) -> tuple[GymnaxEnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: Key, env_state: GymnaxEnvironmentState, action: Array
    ) -> tuple[GymnaxEnvironmentState, Timestep]:
        gymnax_state = env_state.gymnax_state
        obs, gymnax_state, reward, done, _ = self.env.step(
            key, gymnax_state, action, self.env_params
        )
        env_state = GymnaxEnvironmentState(gymnax_state=gymnax_state)
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=reward,
            terminated=done,
            truncated=jnp.bool(False),
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
