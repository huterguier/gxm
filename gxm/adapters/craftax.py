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
class CraftaxState(EnvironmentState):
    craftax_state: Any


class CraftaxAdapter(Environment[CraftaxState]):
    craftax_id: str
    env: Any
    env_params: Any

    def __init__(self, id: str, **kwargs):
        self.craftax_id = id
        self.id = f"Craftax/{id}"
        self.env = make_craftax_env_from_name(id, auto_reset=True, **kwargs)
        self.env_params = self.env.default_params
        self.action_space = _craftax_to_gxm_space(self.env.action_space(self.env_params))
        self.observation_space = _craftax_to_gxm_space(self.env.observation_space(self.env_params))

    def init(self, key: Key) -> tuple[CraftaxState, Timestep]:
        obs, craftax_state = self.env.reset(key, self.env_params)
        obs, _, _, _, info = self.env.step(key, craftax_state, jnp.array(0), self.env_params)
        env_state = CraftaxState(craftax_state=craftax_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=info,
        )
        return env_state, timestep

    def reset(self, key: Key, env_state: CraftaxState) -> tuple[CraftaxState, Timestep]:
        del env_state
        return self.init(key)

    def step(self, key: Key, env_state: CraftaxState, action: Action) -> tuple[CraftaxState, Timestep]:
        craftax_state = env_state.craftax_state
        obs, craftax_state, reward, done, info = self.env.step(
            key, craftax_state, action, self.env_params
        )
        env_state = CraftaxState(craftax_state=craftax_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            reward=reward,
            terminated=done,
            truncated=done,
            info=info,
        )
        return env_state, timestep


def _craftax_to_gxm_space(gymnax_space) -> Space:
    if isinstance(gymnax_space, gymnax.environments.spaces.Discrete):
        return Discrete(gymnax_space.n)
    if isinstance(gymnax_space, gymnax.environments.spaces.Box):
        return Box(low=gymnax_space.low, high=gymnax_space.high, shape=gymnax_space.shape)
    if isinstance(gymnax_space, gymnax.environments.spaces.Dict):
        return Tree({k: _craftax_to_gxm_space(v) for k, v in gymnax_space.spaces.items()})
    if isinstance(gymnax_space, gymnax.environments.spaces.Tuple):
        return Tree([_craftax_to_gxm_space(s) for s in gymnax_space.spaces])
    raise NotImplementedError(f"Craftax space type {type(gymnax_space)} not supported.")


def make(id: str, **kwargs) -> Environment:
    return CraftaxAdapter(id, **kwargs)
