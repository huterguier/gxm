from dataclasses import dataclass

import jax
import jax.numpy as jnp
import navix

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space
from gxm.typing import Action, Key


@jax.tree_util.register_dataclass
@dataclass
class NavixState(EnvironmentState):
    navix_state: navix.Timestep


class NavixAdapter(Environment[NavixState]):
    navix_id: str
    env: navix.Environment

    def __init__(self, id: str, **kwargs):
        self.navix_id = id
        self.id = f"Navix/{id}"
        self.env = navix.make(id, **kwargs)
        self.action_space = _navix_to_gxm_space(self.env.action_space)
        self.observation_space = _navix_to_gxm_space(self.env.observation_space)

    def init(self, key: Key) -> tuple[NavixState, Timestep]:
        navix_state = self.env.reset(key)
        env_state = NavixState(navix_state=navix_state)
        timestep = Timestep(
            next_obs=navix_state.observation,
            true_next_obs=navix_state.observation,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(self, key: Key, env_state: NavixState) -> tuple[NavixState, Timestep]:
        del env_state
        return self.init(key)

    def step(self, key: Key, env_state: NavixState, action: Action) -> tuple[NavixState, Timestep]:
        del key
        navix_state = self.env.step(env_state.navix_state, action)
        env_state = NavixState(navix_state=navix_state)
        timestep = Timestep(
            next_obs=navix_state.observation,
            true_next_obs=navix_state.observation,
            reward=navix_state.reward,
            terminated=navix_state.is_done(),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    @property
    def num_actions(self) -> int:
        return len(self.env.action_set)


def _navix_to_gxm_space(navix_space) -> Space:
    if isinstance(navix_space, navix.spaces.Discrete):
        return Discrete(int(navix_space.n))
    if isinstance(navix_space, navix.spaces.Continuous):
        return Box(low=navix_space.minimum, high=navix_space.maximum, shape=navix_space.shape)
    raise NotImplementedError(f"Navix space type {type(navix_space)} not supported.")


def make(id: str, **kwargs) -> Environment:
    return NavixAdapter(id, **kwargs)
