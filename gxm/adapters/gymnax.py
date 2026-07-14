from dataclasses import dataclass
from typing import Any

import gymnax
import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key
from gxm.wrappers import AutoReset


def _gymnax_to_gxm_space(gymnax_space) -> Space:
    if isinstance(gymnax_space, gymnax_spaces.Discrete):
        return Discrete(gymnax_space.n)
    if isinstance(gymnax_space, gymnax_spaces.Box):
        return Box(
            low=gymnax_space.low, high=gymnax_space.high, shape=gymnax_space.shape
        )
    if isinstance(gymnax_space, gymnax_spaces.Dict):
        return Tree(
            {k: _gymnax_to_gxm_space(v) for k, v in gymnax_space.spaces.items()}
        )
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
            return gymnax_spaces.Dict(
                {k: _gxm_to_gymnax_space(v) for k, v in space.spaces.items()}
            )
    raise NotImplementedError(f"Gxm space type {type(space)} not supported.")


@jax.tree_util.register_dataclass
@dataclass
class GymnaxState(EnvironmentState):
    gymnax_state: gymnax.EnvState


class GymnaxAdapter(Environment[GymnaxState]):
    """
    Adapter over a gymnax environment.

    Uses gymnax's non-resetting ``step_env`` rather than ``step``: gymnax's own
    ``step`` auto-resets internally, which destroys the terminal observation and
    makes truncation unrecoverable. Consequently this adapter does *not*
    auto-reset — stepping past ``done`` is undefined. ``make`` wraps it in
    :class:`gxm.wrappers.AutoReset` by default.

    Truncation is recovered from the state's step counter: a ``done`` step with
    ``state.time >= params.max_steps_in_episode`` is labeled truncated (and only
    truncated, even if the environment also reached a terminal state on that
    exact step — the coincidence is not distinguishable through gymnax's API).
    """

    gymnax_id: str
    env: gymnax.environments.environment.Environment
    env_params: Any

    def __init__(self, id: str, **kwargs):
        self.gymnax_id = id
        self.id = f"Gymnax/{id}"
        self.env, self.env_params = gymnax.make(id, **kwargs)
        self.action_space = _gymnax_to_gxm_space(self.env.action_space(self.env_params))
        self.observation_space = _gymnax_to_gxm_space(
            self.env.observation_space(self.env_params)
        )
        # Probe one step eagerly to learn the info structure (so init can
        # return a structurally identical, zeroed info) and whether the
        # environment carries a step counter for truncation recovery.
        key_probe = jax.random.key(0)
        _, state_probe = self.env.reset(key_probe, self.env_params)
        action_probe = self.env.action_space(self.env_params).sample(key_probe)
        *_, info_probe = self.env.step_env(
            key_probe, state_probe, action_probe, self.env_params
        )
        self._init_info = jax.tree.map(jnp.zeros_like, info_probe)
        self._time_limited = hasattr(state_probe, "time") and hasattr(
            self.env_params, "max_steps_in_episode"
        )

    def init(self, key: Key) -> tuple[GymnaxState, Timestep]:
        obs, gymnax_state = self.env.reset(key, self.env_params)
        state = GymnaxState(gymnax_state=gymnax_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=jax.tree.map(jnp.zeros_like, self.action_space.sample(key)),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=self._init_info,
        )
        return state, timestep

    def reset(self, key: Key, state: GymnaxState) -> tuple[GymnaxState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: GymnaxState, action: Array
    ) -> tuple[GymnaxState, Timestep]:
        gymnax_state = state.gymnax_state
        obs, gymnax_state, reward, done, info = self.env.step_env(
            key, gymnax_state, action, self.env_params
        )
        done = jnp.asarray(done)
        if self._time_limited:
            truncated = jnp.logical_and(
                done, gymnax_state.time >= self.env_params.max_steps_in_episode
            )
        else:
            truncated = jnp.zeros_like(done)
        state = GymnaxState(gymnax_state=gymnax_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=action,
            reward=reward,
            terminated=jnp.logical_and(done, jnp.logical_not(truncated)),
            truncated=truncated,
            info=info,
        )
        return state, timestep


@jax.tree_util.register_dataclass
@dataclass
class _WrappedGymnaxState(EnvironmentState):
    gymnax_state: Any


class _GymnaxToGxm(Environment[_WrappedGymnaxState]):
    """Non-resetting gxm view of an existing gymnax environment object.

    Same semantics as :class:`GymnaxAdapter` (``step_env``-based, truncation
    recovered from the step counter); ``wrap`` adds :class:`AutoReset` on top
    by default.
    """

    def __init__(self, env: Any, params: Any = None):
        self._env = env
        self._params = params if params is not None else env.default_params
        self.id = "gymnax_wrapped"
        self.action_space = _gymnax_to_gxm_space(self._env.action_space(self._params))
        self.observation_space = _gymnax_to_gxm_space(
            self._env.observation_space(self._params)
        )
        key_probe = jax.random.key(0)
        _, state_probe = self._env.reset(key_probe, self._params)
        action_probe = self._env.action_space(self._params).sample(key_probe)
        *_, info_probe = self._env.step_env(
            key_probe, state_probe, action_probe, self._params
        )
        self._init_info = jax.tree.map(jnp.zeros_like, info_probe)
        self._time_limited = hasattr(state_probe, "time") and hasattr(
            self._params, "max_steps_in_episode"
        )

    def init(self, key: Key) -> tuple[_WrappedGymnaxState, Timestep]:
        obs, gymnax_state = self._env.reset(key, self._params)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=jax.tree.map(jnp.zeros_like, self.action_space.sample(key)),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=self._init_info,
        )
        return _WrappedGymnaxState(gymnax_state=gymnax_state), timestep

    def reset(
        self, key: Key, state: _WrappedGymnaxState
    ) -> tuple[_WrappedGymnaxState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: _WrappedGymnaxState, action: Array
    ) -> tuple[_WrappedGymnaxState, Timestep]:
        obs, gymnax_state, reward, done, info = self._env.step_env(
            key, state.gymnax_state, action, self._params
        )
        done = jnp.asarray(done)
        if self._time_limited:
            truncated = jnp.logical_and(
                done, gymnax_state.time >= self._params.max_steps_in_episode
            )
        else:
            truncated = jnp.zeros_like(done)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=action,
            reward=reward,
            terminated=jnp.logical_and(done, jnp.logical_not(truncated)),
            truncated=truncated,
            info=info,
        )
        return _WrappedGymnaxState(gymnax_state=gymnax_state), timestep


class _GxmToGymnax:
    def __init__(self, env: Environment):
        self._env = env

    @property
    def default_params(self) -> None:
        return None

    def step(self, key: Key, state: Any, action: Any, params: Any | None = None):
        del params
        next_state, timestep = self._env.step(key, state, action)
        return (
            timestep.next_obs,
            next_state,
            timestep.reward,
            timestep.done,
            timestep.info,
        )

    def reset(self, key: Key, params: Any | None = None):
        del params
        state, timestep = self._env.init(key)
        return timestep.next_obs, state

    def action_space(self, params: Any | None = None):
        del params
        return _gxm_to_gymnax_space(self._env.action_space)

    def observation_space(self, params: Any | None = None):
        del params
        return _gxm_to_gymnax_space(self._env.observation_space)


def make(id: str, autoreset: bool = True, **kwargs) -> Environment:
    env = GymnaxAdapter(id, **kwargs)
    return AutoReset(env).seal() if autoreset else env


def wrap(env: Any, params: Any = None, autoreset: bool = True) -> Environment:
    """Wrap a gymnax environment object as a gxm environment."""
    wrapped = _GymnaxToGxm(env, params)
    return AutoReset(wrapped).seal() if autoreset else wrapped


def unwrap(env: Environment) -> _GxmToGymnax:
    """Wrap a gxm environment as a gymnax environment."""
    return _GxmToGymnax(env)
