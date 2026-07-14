from dataclasses import dataclass
from typing import Any

import gymnax.environments.spaces
import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Action, Key
from gxm.wrappers import AutoReset


@jax.tree_util.register_dataclass
@dataclass
class CraftaxState(EnvironmentState):
    craftax_state: Any


class CraftaxAdapter(Environment[CraftaxState]):
    """
    Adapter over a craftax environment.

    Uses craftax's ``NoAutoReset`` variants (craftax's own auto-reset destroys
    the terminal observation), so this adapter does *not* auto-reset — ``make``
    wraps it in :class:`gxm.wrappers.AutoReset` by default.

    Truncation is recovered from the state's step counter: a ``done`` step with
    ``state.timestep >= params.max_timesteps`` is labeled truncated (and only
    truncated, even if the agent also died on that exact step — the coincidence
    is not distinguishable through craftax's API).

    Note on cost: :class:`AutoReset` computes a fresh world every step, matching
    the cost of craftax's own per-environment auto-reset. Craftax's *optimistic*
    resetting (sharing few generated worlds across a large batch) is a
    batch-level strategy that does not fit the per-environment wrapper model
    and is not yet supported.
    """

    craftax_id: str
    env: Any
    env_params: Any

    def __init__(self, id: str):
        self.craftax_id = id
        self.id = f"Craftax/{id}"
        self.env = make_craftax_env_from_name(id, auto_reset=False)
        self.env_params = self.env.default_params
        self.action_space = _craftax_to_gxm_space(
            self.env.action_space(self.env_params)
        )
        self.observation_space = _craftax_to_gxm_space(
            self.env.observation_space(self.env_params)
        )
        # Probe one step eagerly so init can return a structurally identical,
        # zeroed info.
        key_probe = jax.random.key(0)
        obs_probe, state_probe = self.env.reset(key_probe, self.env_params)
        action_probe = self.env.action_space(self.env_params).sample(key_probe)
        *_, info_probe = self.env.step(
            key_probe, state_probe, action_probe, self.env_params
        )
        self._init_info = jax.tree.map(jnp.zeros_like, info_probe)
        # Craftax's pixel envs declare a transposed (W, H, C) observation
        # space while actually emitting (H, W, C); trust the actual
        # observation shape.
        if (
            isinstance(self.observation_space, Box)
            and self.observation_space.shape != jnp.shape(obs_probe)
        ):
            self.observation_space = Box(
                low=jnp.min(self.observation_space.low),
                high=jnp.max(self.observation_space.high),
                shape=jnp.shape(obs_probe),
            )

    def init(self, key: Key) -> tuple[CraftaxState, Timestep]:
        obs, craftax_state = self.env.reset(key, self.env_params)
        state = CraftaxState(craftax_state=craftax_state)
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

    def reset(self, key: Key, state: CraftaxState) -> tuple[CraftaxState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: CraftaxState, action: Action
    ) -> tuple[CraftaxState, Timestep]:
        obs, craftax_state, reward, done, info = self.env.step(
            key, state.craftax_state, action, self.env_params
        )
        done = jnp.asarray(done)
        truncated = jnp.logical_and(
            done, craftax_state.timestep >= self.env_params.max_timesteps
        )
        state = CraftaxState(craftax_state=craftax_state)
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


def _craftax_to_gxm_space(gymnax_space) -> Space:
    if isinstance(gymnax_space, gymnax.environments.spaces.Discrete):
        return Discrete(gymnax_space.n)
    if isinstance(gymnax_space, gymnax.environments.spaces.Box):
        return Box(
            low=gymnax_space.low, high=gymnax_space.high, shape=gymnax_space.shape
        )
    if isinstance(gymnax_space, gymnax.environments.spaces.Dict):
        return Tree(
            {k: _craftax_to_gxm_space(v) for k, v in gymnax_space.spaces.items()}
        )
    if isinstance(gymnax_space, gymnax.environments.spaces.Tuple):
        return Tree([_craftax_to_gxm_space(s) for s in gymnax_space.spaces])
    raise NotImplementedError(f"Craftax space type {type(gymnax_space)} not supported.")


def make(id: str, autoreset: bool = True, **kwargs) -> Environment:
    env = CraftaxAdapter(id, **kwargs)
    return AutoReset(env).seal() if autoreset else env
