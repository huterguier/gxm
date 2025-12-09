from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.typing import Action, Key


class GxmEnvironmentState(EnvironmentState):
    pass


class GxmEnvironment(Environment[GxmEnvironmentState], ABC):
    """Base class for Gxm environments."""

    gxm_id: str
    """ The Gxm environment ID. """

    def init(self, key: Key) -> tuple[GxmEnvironmentState, Timestep]:
        env_state, timestep = self._reset(key)
        return env_state, timestep

    def reset(
        self, key: Key, env_state: GxmEnvironmentState
    ) -> tuple[GxmEnvironmentState, Timestep]:
        env_state, timestep = self._reset(key)
        return env_state, timestep

    @abstractmethod
    def _reset(self, key: Key) -> tuple[GxmEnvironmentState, Timestep]:
        pass

    def step(
        self, key: Key, env_state: GxmEnvironmentState, action: Action
    ) -> tuple[GxmEnvironmentState, Timestep]:
        env_state_step, timestep_step = self._step(key, env_state, action)
        env_state_reset, timestep_reset = self._reset(key)
        env_state = jax.tree.map(
            lambda x_step, x_reset: jnp.where(timestep_step.done, x_reset, x_step),
            env_state_step,
            env_state_reset,
        )
        true_obs = jax.tree.map(
            lambda x_step, x_reset: jnp.where(timestep_step.truncated, x_reset, x_step),
            timestep_step.obs,
            timestep_reset.obs,
        )
        timestep = Timestep(
            obs=timestep_step.obs,
            true_obs=true_obs,
            reward=timestep_step.reward,
            terminated=timestep_step.terminated,
            truncated=timestep_step.truncated,
            info=timestep_step.info,
        )
        return env_state, timestep

    @abstractmethod
    def _step(
        self, key: Key, env_state: GxmEnvironmentState, action: Action
    ) -> tuple[GxmEnvironmentState, Timestep]:
        pass
