from dataclasses import dataclass
from typing import Any

import brax
import brax.envs
import jax
import jax.numpy as jnp
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box
from gxm.typing import Array, Key


@jax.tree_util.register_dataclass
@dataclass
class BraxEnvironmentState(EnvironmentState):
    """State for Brax environments."""

    brax_state: brax.envs.State
    """ The Brax environment state. """


class BraxEnvironment(Environment):
    """Base class for Brax environments."""

    brax_id: str
    """ The Brax environment ID. """
    env: brax.envs.Env
    """ The Brax environment. """

    def __init__(self, id: str, **kwargs):

        self.id = id
        self.brax_id = id.split("/", 1)[1]
        self.env = brax.envs.create(self.brax_id, **kwargs)
        self.env = EpisodeWrapper(self.env, episode_length=1000, action_repeat=1)
        self.env = AutoResetWrapper(self.env)
        self.action_space = Box(
            low=jnp.full((self.env.action_size,), -1.0),
            high=jnp.full((self.env.action_size,), 1.0),
            shape=(self.env.action_size,),
        )
        self.observation_space = Box(
            low=jnp.full((self.env.observation_size,), -jnp.inf),
            high=jnp.full((self.env.observation_size,), jnp.inf),
            shape=(self.env.observation_size,),
        )

    def init(self, key: Key) -> tuple[BraxEnvironmentState, Timestep]:
        brax_state = self.env.reset(key)
        env_state = BraxEnvironmentState(brax_state=brax_state)
        timestep = Timestep(
            obs=brax_state.obs,
            true_obs=brax_state.obs,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(
        self, key: Key, env_state: BraxEnvironmentState
    ) -> tuple[BraxEnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: Key, env_state: BraxEnvironmentState, action: Array
    ) -> tuple[BraxEnvironmentState, Timestep]:
        del key
        brax_state = self.env.step(env_state.brax_state, action)
        env_state = BraxEnvironmentState(brax_state=brax_state)
        timestep = Timestep(
            obs=brax_state.obs,
            true_obs=brax_state.obs,
            reward=brax_state.reward,
            terminated=brax_state.done > 0.5,
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep
