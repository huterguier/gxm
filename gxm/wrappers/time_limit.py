from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import EnvironmentWrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class TimeLimitState(WrapperState):
    time: Array


class TimeLimit(EnvironmentWrapper[TimeLimitState]):
    """
    Wrapper that terminates an episode after a fixed number of steps.
    """

    wrapped: Environment

    def __init__(self, wrapped: Environment, time_limit: int = 1000):
        """
        Args:
            wrapped: The environment to wrap.
            time_limit: Maximum number of steps before the episode is truncated.
        """
        super().__init__(wrapped)
        self.time_limit = time_limit

    def init(self, key: Key) -> tuple[TimeLimitState, Timestep]:
        wrapped_state, timestep = self.wrapped.init(key)
        time_limit_state = TimeLimitState(
            wrapped_state=wrapped_state,
            time=jnp.array(0, dtype=jnp.int32),
        )
        return time_limit_state, timestep

    def reset(self, key: Key, state: TimeLimitState) -> tuple[TimeLimitState, Timestep]:
        wrapped_state, timestep = self.wrapped.reset(key, state.wrapped_state)
        time_limit_state = TimeLimitState(
            wrapped_state=wrapped_state,
            time=jnp.array(0, dtype=jnp.int32),
        )
        return time_limit_state, timestep

    def step(
        self,
        key: Key,
        state: TimeLimitState,
        action: PyTree,
    ) -> tuple[TimeLimitState, Timestep]:
        key_step, key_reset = jax.random.split(key)
        step_wrapped_state, timestep = self.wrapped.step(
            key_step, state.wrapped_state, action
        )
        time_limit_state = TimeLimitState(
            wrapped_state=step_wrapped_state,
            time=state.time + 1,
        )
        reset_wrapped_state, reset_timestep = self.wrapped.reset(
            key_reset, state.wrapped_state
        )
        reset_time_limit_state = TimeLimitState(
            wrapped_state=reset_wrapped_state,
            time=jnp.array(0, dtype=jnp.int32),
        )
        reset_timestep = Timestep(
            next_obs=reset_timestep.next_obs,
            true_next_obs=timestep.next_obs,
            action=action,
            reward=timestep.reward,
            terminated=jnp.array(False, dtype=jnp.bool),
            truncated=jnp.array(True, dtype=jnp.bool),
            info=timestep.info,
        )
        time_limit_state, timestep = jax.lax.cond(
            jnp.logical_and(state.time + 1 >= self.time_limit, ~timestep.done),
            lambda: (reset_time_limit_state, reset_timestep),
            lambda: (time_limit_state, timestep),
        )
        return time_limit_state, timestep
