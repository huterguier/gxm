from typing import Any, Optional

import gymnax.environments.spaces as gymnax_spaces
import jax.numpy as jnp

from gxm.core import Environment
from gxm.spaces import Box, Discrete, Tree
from gxm.typing import Array, Key


class GxmToGymnax:
    """
    Wrapper that converts a Gxm environment to a Gymnax environment.
    """

    def __init__(self, env: Environment):
        self._env = env

    @property
    def default_params(self) -> None:
        return None

    def step(
        self,
        key: Key,
        state: Any,
        action: Any,
        params: Optional[Any] = None,
    ) -> tuple[Any, Any, Array, Array, dict[str, Any]]:
        del params
        next_state, timestep = self._env.step(key, state, action)
        return (
            timestep.obs,
            next_state,
            timestep.reward,
            timestep.done,
            timestep.info,
        )

    def reset(self, key: Key, params: Optional[Any] = None) -> tuple[Any, Any]:
        del params
        state, timestep = self._env.init(key)
        return timestep.obs, state

    def action_space(self, params: Optional[Any] = None) -> gymnax_spaces.Space:
        del params
        return self._gxm_to_gymnax_space(self._env.action_space)

    def observation_space(self, params: Optional[Any] = None) -> gymnax_spaces.Space:
        del params
        return self._gxm_to_gymnax_space(self._env.observation_space)

    def _gxm_to_gymnax_space(self, space: Any) -> gymnax_spaces.Space:
        if isinstance(space, Discrete):
            return gymnax_spaces.Discrete(space.n)
        if isinstance(space, Box):
            return gymnax_spaces.Box(space.low, space.high, space.shape, jnp.float32)
        if isinstance(space, Tree):
            if isinstance(space.spaces, (list, tuple)):
                return gymnax_spaces.Tuple(
                    [self._gxm_to_gymnax_space(s) for s in space.spaces]
                )
            if isinstance(space.spaces, dict):
                return gymnax_spaces.Dict(
                    {k: self._gxm_to_gymnax_space(v) for k, v in space.spaces.items()}
                )
        raise NotImplementedError(f"Gxm space type {type(space)} not supported.")
