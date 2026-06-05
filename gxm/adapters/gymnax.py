from dataclasses import dataclass
from typing import Any, Optional

import gymnax
import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key


def _gymnax_to_gxm_space(gymnax_space) -> Space:
    if isinstance(gymnax_space, gymnax_spaces.Discrete):
        return Discrete(gymnax_space.n)
    if isinstance(gymnax_space, gymnax_spaces.Box):
        return Box(low=gymnax_space.low, high=gymnax_space.high, shape=gymnax_space.shape)
    if isinstance(gymnax_space, gymnax_spaces.Dict):
        return Tree({k: _gymnax_to_gxm_space(v) for k, v in gymnax_space.spaces.items()})
    if isinstance(gymnax_space, gymnax_spaces.Tuple):
        return Tree([_gymnax_to_gxm_space(s) for s in gymnax_space.spaces])
    raise NotImplementedError(f"Gymnax space type {type(gymnax_space)} not supported.")


def _gxm_to_gymnax_space(space: Space):
    if isinstance(space, Discrete):
        return gymnax_spaces.Discrete(space.n)
    if isinstance(space, Box):
        return gymnax_spaces.Box(space.low, space.high, space.shape, jnp.float32)
    if isinstance(space, Tree):
        if isinstance(space.spaces, (list, tuple)):
            return gymnax_spaces.Tuple([_gxm_to_gymnax_space(s) for s in space.spaces])
        if isinstance(space.spaces, dict):
            return gymnax_spaces.Dict({k: _gxm_to_gymnax_space(v) for k, v in space.spaces.items()})
    raise NotImplementedError(f"Gxm space type {type(space)} not supported.")


@jax.tree_util.register_dataclass
@dataclass
class GymnaxState(EnvironmentState):
    gymnax_state: gymnax.EnvState


class GymnaxAdapter(Environment[GymnaxState]):
    gymnax_id: str
    env: gymnax.environments.environment.Environment
    env_params: Any

    def __init__(self, id: str, **kwargs):
        self.gymnax_id = id
        self.id = f"Gymnax/{id}"
        self.env, self.env_params = gymnax.make(id, **kwargs)
        self.action_space = _gymnax_to_gxm_space(self.env.action_space(self.env_params))
        self.observation_space = _gymnax_to_gxm_space(self.env.observation_space(self.env_params))

    def init(self, key: Key) -> tuple[GymnaxState, Timestep]:
        obs, gymnax_state = self.env.reset(key, self.env_params)
        env_state = GymnaxState(gymnax_state=gymnax_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=self.action_space.sample(key),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(self, key: Key, env_state: GymnaxState) -> tuple[GymnaxState, Timestep]:
        del env_state
        return self.init(key)

    def step(self, key: Key, env_state: GymnaxState, action: Array) -> tuple[GymnaxState, Timestep]:
        gymnax_state = env_state.gymnax_state
        obs, gymnax_state, reward, done, _ = self.env.step(key, gymnax_state, action, self.env_params)
        env_state = GymnaxState(gymnax_state=gymnax_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=action,
            reward=reward,
            terminated=done,
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep


@jax.tree_util.register_dataclass
@dataclass
class _WrappedGymnaxState(EnvironmentState):
    gymnax_state: Any


class _GymnaxToGxm(Environment[_WrappedGymnaxState]):
    def __init__(self, env: Any, params: Any = None):
        self._env = env
        self._params = params if params is not None else env.default_params
        self.id = "gymnax_wrapped"
        self.action_space = _gymnax_to_gxm_space(self._env.action_space(self._params))
        self.observation_space = _gymnax_to_gxm_space(self._env.observation_space(self._params))

    def init(self, key: Key) -> tuple[_WrappedGymnaxState, Timestep]:
        obs, state = self._env.reset(key, self._params)
        sentinel_action = self._env.action_space(self._params).sample(key)
        _, _, _, _, info = self._env.step(key, state, sentinel_action, self._params)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=self.action_space.sample(key),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=info,
        )
        return _WrappedGymnaxState(gymnax_state=state), timestep

    def reset(self, key: Key, env_state: _WrappedGymnaxState) -> tuple[_WrappedGymnaxState, Timestep]:
        del env_state
        return self.init(key)

    def step(self, key: Key, env_state: _WrappedGymnaxState, action: Array) -> tuple[_WrappedGymnaxState, Timestep]:
        obs, state, reward, done, info = self._env.step(
            key, env_state.gymnax_state, action, self._params
        )
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=action,
            reward=reward,
            terminated=done,
            truncated=jnp.bool(False),
            info=info,
        )
        return _WrappedGymnaxState(gymnax_state=state), timestep


class _GxmToGymnax:
    def __init__(self, env: Environment):
        self._env = env

    @property
    def default_params(self) -> None:
        return None

    def step(self, key: Key, state: Any, action: Any, params: Optional[Any] = None):
        del params
        next_state, timestep = self._env.step(key, state, action)
        return timestep.next_obs, next_state, timestep.reward, timestep.done, timestep.info

    def reset(self, key: Key, params: Optional[Any] = None):
        del params
        state, timestep = self._env.init(key)
        return timestep.next_obs, state

    def action_space(self, params: Optional[Any] = None):
        del params
        return _gxm_to_gymnax_space(self._env.action_space)

    def observation_space(self, params: Optional[Any] = None):
        del params
        return _gxm_to_gymnax_space(self._env.observation_space)


def make(id: str, **kwargs) -> Environment:
    return GymnaxAdapter(id, **kwargs)


def wrap(env: Any, params: Any = None) -> Environment:
    """Wrap a gymnax environment object as a gxm environment."""
    return _GymnaxToGxm(env, params)


def unwrap(env: Environment) -> _GxmToGymnax:
    """Wrap a gxm environment as a gymnax environment."""
    return _GxmToGymnax(env)
