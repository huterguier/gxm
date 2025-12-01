from dataclasses import dataclass

import jax
import jax.numpy as jnp
import navix

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space
from gxm.typing import Action, Key


@jax.tree_util.register_dataclass
@dataclass
class NavixEnvironmentState(EnvironmentState):
    """State for Navix environments."""

    navix_state: navix.Timestep
    """The Navix state."""


class NavixEnvironment(Environment):
    """Base class for Gymnax environments."""

    navix_id: str
    """The Navix environment ID."""
    env: navix.Environment
    """The Navix environment instance."""

    def __init__(self, id: str, **kwargs):
        self.id = id
        self.navix_id = id.split("/", 1)[1]
        self.env = navix.make(self.navix_id, **kwargs)
        self.action_space = self.navix_to_gxm_space(self.env.action_space)
        self.observation_space = self.navix_to_gxm_space(self.env.observation_space)

    def init(self, key: Key) -> tuple[NavixEnvironmentState, Timestep]:
        navix_state = self.env.reset(key)
        env_state = NavixEnvironmentState(navix_state=navix_state)
        timestep = Timestep(
            obs=navix_state.observation,
            true_obs=navix_state.observation,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(
        self, key: Key, env_state: NavixEnvironmentState
    ) -> tuple[NavixEnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: Key, env_state: NavixEnvironmentState, action: Action
    ) -> tuple[NavixEnvironmentState, Timestep]:
        del key
        navix_state = env_state.navix_state
        navix_state = self.env.step(navix_state, action)
        env_state = NavixEnvironmentState(navix_state=navix_state)
        timestep = Timestep(
            obs=navix_state.observation,
            true_obs=navix_state.observation,
            reward=navix_state.reward,
            terminated=navix_state.is_done(),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    @property
    def num_actions(self) -> int:
        return len(self.env.action_set)

    @classmethod
    def navix_to_gxm_space(cls, navix_space) -> Space:
        """Convert a Navix space to a Gxm space."""
        if isinstance(navix_space, navix.spaces.Discrete):
            return Discrete(int(navix_space.n))
        if isinstance(navix_space, navix.spaces.Continuous):
            return Box(
                low=navix_space.minimum,
                high=navix_space.maximum,
                shape=navix_space.shape,
            )
        else:
            raise NotImplementedError(
                f"Gymnax space type {type(navix_space)} not supported."
            )
