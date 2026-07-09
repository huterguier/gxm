from typing import Any

import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.typing import Key, PyTree
from gxm.wrappers.wrapper import EnvironmentWrapper


class AutoReset(EnvironmentWrapper[Any]):
    """
    Wrapper that automatically resets an environment on episode end.

    On each step, both the stepped state and a freshly reset state are computed
    and blended via ``jnp.where`` on ``done``, keeping output shapes static under
    ``jit``/``vmap``/``scan``.
    """

    def __init__(self, wrapped: Environment, unwrap: bool = True):
        super().__init__(wrapped, unwrap=unwrap)

    def init(self, key: Key) -> tuple[EnvironmentState, Timestep]:
        return self.wrapped.init(key)

    def reset(
        self, key: Key, state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        return self.wrapped.reset(key, state)

    def step(
        self, key: Key, state: EnvironmentState, action: PyTree
    ) -> tuple[EnvironmentState, Timestep]:
        key_step, key_reset = jax.random.split(key)
        state_step, timestep_step = self.wrapped.step(key_step, state, action)
        state_reset, timestep_reset = self.wrapped.reset(key_reset, state)
        state = jax.tree.map(
            lambda x_step, x_reset: jnp.where(timestep_step.done, x_reset, x_step),
            state_step,
            state_reset,
        )
        obs = jax.tree.map(
            lambda x_step, x_reset: jnp.where(timestep_step.done, x_reset, x_step),
            timestep_step.next_obs,
            timestep_reset.next_obs,
        )
        true_obs = jax.tree.map(
            lambda x_step, x_obs: jnp.where(timestep_step.truncated, x_step, x_obs),
            timestep_step.next_obs,
            obs,
        )
        return state, Timestep(
            next_obs=obs,
            true_next_obs=true_obs,
            action=action,
            reward=timestep_step.reward,
            terminated=timestep_step.terminated,
            truncated=timestep_step.truncated,
            info=timestep_step.info,
        )
