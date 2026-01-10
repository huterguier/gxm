from dataclasses import dataclass
from typing import Any

import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key


@jax.tree_util.register_dataclass
@dataclass
class GymnaxToGxmState(EnvironmentState):
    gymnax_state: Any


class GymnaxToGxm(Environment):
    """
    Wrapper that converts a Gymnax environment to a Gxm environment.
    """

    def __init__(self, env: Any, params: Any = None):
        self._env = env
        self._params = params if params is not None else env.default_params
        self.id = "gymnax_wrapped"

        self.action_space = self._gymnax_to_gxm_space(
            self._env.action_space(self._params)
        )
        self.observation_space = self._gymnax_to_gxm_space(
            self._env.observation_space(self._params)
        )

    def init(self, key: Key) -> tuple[GymnaxToGxmState, Timestep]:
        obs, state = self._env.reset(key, self._params)
        _, _, _, _, info = self._env.step(
            key, state, self._env.action_space(self._params).sample(key), self._params
        )
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=info,
        )
        return GymnaxToGxmState(gymnax_state=state), timestep

    def reset(
        self, key: Key, env_state: GymnaxToGxmState
    ) -> tuple[GymnaxToGxmState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: Key, env_state: GymnaxToGxmState, action: Array
    ) -> tuple[GymnaxToGxmState, Timestep]:
        obs, state, reward, done, info = self._env.step(
            key, env_state.gymnax_state, action, self._params
        )
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=reward,
            terminated=done,
            truncated=jnp.bool(False),
            info=info,
        )
        return GymnaxToGxmState(gymnax_state=state), timestep

    @classmethod
    def _gymnax_to_gxm_space(cls, gymnax_space: Any) -> Space:
        if isinstance(gymnax_space, gymnax_spaces.Discrete):
            return Discrete(gymnax_space.n)
        if isinstance(gymnax_space, gymnax_spaces.Box):
            return Box(
                low=gymnax_space.low,
                high=gymnax_space.high,
                shape=gymnax_space.shape,
            )
        if isinstance(gymnax_space, gymnax_spaces.Dict):
            return Tree(
                {k: cls._gymnax_to_gxm_space(v) for k, v in gymnax_space.spaces.items()}
            )
        if isinstance(gymnax_space, gymnax_spaces.Tuple):
            return Tree([cls._gymnax_to_gxm_space(s) for s in gymnax_space.spaces])

        raise NotImplementedError(
            f"Gymnax space type {type(gymnax_space)} not supported."
        )
