from dataclasses import dataclass
from typing import Any

import gymnax.environments.spaces
import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Action, Key


@jax.tree_util.register_dataclass
@dataclass
class CraftaxEnvironmentState(EnvironmentState):
    """State for Craftax environments."""

    craftax_state: Any
    """ The Craftax environment state. """


class CraftaxEnvironment(Environment[CraftaxEnvironmentState]):
    """Base class for Craftax environments."""

    craftax_id: str
    """ The Craftax environment ID. """
    env: Any
    """ The Craftax environment. """
    env_params = Any
    """ The parameters for the Craftax environment. """

    def __init__(self, id: str, **kwargs):
        self.id = id
        self.craftax_id = id.split("/", 1)[1]
        self.env = make_craftax_env_from_name(
            self.craftax_id, auto_reset=True, **kwargs
        )
        self.env_params = self.env.default_params
        self.action_space = self.craftax_to_gxm_space(
            self.env.action_space(self.env_params)
        )
        self.observation_space = self.craftax_to_gxm_space(
            self.env.observation_space(self.env_params)
        )

    def init(self, key: Key) -> tuple[CraftaxEnvironmentState, Timestep]:
        obs, craftax_state = self.env.reset(key, self.env_params)
        obs, _, _, _, info = self.env.step(
            key, craftax_state, jnp.array(0), self.env_params
        )
        env_state = CraftaxEnvironmentState(craftax_state=craftax_state)
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=info,
        )
        return env_state, timestep

    def reset(
        self, key: Key, env_state: CraftaxEnvironmentState
    ) -> tuple[CraftaxEnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: Key, env_state: CraftaxEnvironmentState, action: Action
    ) -> tuple[CraftaxEnvironmentState, Timestep]:
        craftax_state = env_state.craftax_state
        obs, craftax_state, reward, done, info = self.env.step(
            key, craftax_state, action, self.env_params
        )
        env_state = CraftaxEnvironmentState(craftax_state=craftax_state)
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=reward,
            terminated=done,
            truncated=done,
            info=info,
        )
        return env_state, timestep

    @classmethod
    def craftax_to_gxm_space(cls, gymnax_space) -> Space:
        """Convert a Craftax space to a Gxm space."""
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
                {k: cls.craftax_to_gxm_space(v) for k, v in gymnax_space.spaces.items()}
            )
        if isinstance(gymnax_space, gymnax.environments.spaces.Tuple):
            return Tree([cls.craftax_to_gxm_space(s) for s in gymnax_space.spaces])
        else:
            raise NotImplementedError(
                f"Craftax space type {type(gymnax_space)} not supported."
            )
