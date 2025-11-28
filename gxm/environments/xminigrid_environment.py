from dataclasses import dataclass

import jax
import jax.numpy as jnp
import xminigrid
import xminigrid.environment

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Discrete, Tree
from gxm.typing import Key


@jax.tree_util.register_dataclass
@dataclass
class XMiniGridEnvironmentState(EnvironmentState):
    """State for XMiniGrid environments."""

    xminigrid_state: xminigrid.environment.TimeStep
    """ The XMiniGrid environment state. """


class XMiniGridEnvironment(Environment[XMiniGridEnvironmentState]):
    """Base class for XMiniGrid environments."""

    xminigrid_id: str
    """The XMiniGrid environment ID."""
    env: xminigrid.environment.Environment
    """The XMiniGrid environment instance."""
    env_params: xminigrid.environment.EnvParams
    """The parameters for the XMiniGrid environment."""

    def __init__(self, id: str, **kwargs):
        self.id = "XMiniGrid/" + id
        self.xminigrid_id = id.split("/", 1)[1]
        self.env, self.env_params = xminigrid.make(self.xminigrid_id, **kwargs)
        self.action_space = Discrete(self.env.num_actions(self.env_params))
        observation_shape = self.env.observation_shape(self.env_params)
        assert type(observation_shape) is tuple
        self.observation_space = Tree(tuple(Discrete(n) for n in observation_shape))

    def init(self, key: Key) -> tuple[XMiniGridEnvironmentState, Timestep]:
        xminigrid_state = self.env.reset(self.env_params, key)
        env_state = XMiniGridEnvironmentState(xminigrid_state=xminigrid_state)
        timestep = Timestep(
            obs=xminigrid_state.observation,
            true_obs=xminigrid_state.observation,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(False),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(
        self, key: jax.Array, env_state: XMiniGridEnvironmentState
    ) -> tuple[XMiniGridEnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: jax.Array, env_state: XMiniGridEnvironmentState, action: jax.Array
    ) -> tuple[XMiniGridEnvironmentState, Timestep]:
        del key
        xminigrid_state = env_state.xminigrid_state
        xminigrid_state = self.env.step(self.env_params, xminigrid_state, action)
        env_state = XMiniGridEnvironmentState(xminigrid_state=xminigrid_state)
        timestep = Timestep(
            obs=xminigrid_state.observation,
            true_obs=xminigrid_state.observation,
            reward=xminigrid_state.reward,
            terminated=xminigrid_state.last(),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep
