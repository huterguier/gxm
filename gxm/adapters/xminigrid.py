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
class XMiniGridState(EnvironmentState):
    xminigrid_state: xminigrid.environment.TimeStep


class XMiniGridAdapter(Environment[XMiniGridState]):
    xminigrid_id: str
    env: xminigrid.environment.Environment
    env_params: xminigrid.environment.EnvParams

    def __init__(self, id: str, **kwargs):
        self.xminigrid_id = id
        self.id = f"XMiniGrid/{id}"
        self.env, self.env_params = xminigrid.make(id, **kwargs)
        self.action_space = Discrete(self.env.num_actions(self.env_params))
        observation_shape = self.env.observation_shape(self.env_params)
        assert type(observation_shape) is tuple
        self.observation_space = Tree(tuple(Discrete(n) for n in observation_shape))

    def init(self, key: Key) -> tuple[XMiniGridState, Timestep]:
        xminigrid_state = self.env.reset(self.env_params, key)
        env_state = XMiniGridState(xminigrid_state=xminigrid_state)
        timestep = Timestep(
            obs=xminigrid_state.observation,
            true_obs=xminigrid_state.observation,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(False),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(self, key: jax.Array, env_state: XMiniGridState) -> tuple[XMiniGridState, Timestep]:
        del env_state
        return self.init(key)

    def step(self, key: jax.Array, env_state: XMiniGridState, action: jax.Array) -> tuple[XMiniGridState, Timestep]:
        del key
        xminigrid_state = self.env.step(self.env_params, env_state.xminigrid_state, action)
        env_state = XMiniGridState(xminigrid_state=xminigrid_state)
        timestep = Timestep(
            obs=xminigrid_state.observation,
            true_obs=xminigrid_state.observation,
            reward=xminigrid_state.reward,
            terminated=xminigrid_state.last(),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep


def make(id: str, **kwargs) -> Environment:
    return XMiniGridAdapter(id, **kwargs)
