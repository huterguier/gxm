from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class TimeLimitState(WrapperState):
    time: Array


class TimeLimit(Wrapper):
    """
    Wrapper that terminates an episode after a fixed number of steps.
    """

    env: Environment

    def __init__(self, env: Environment, unwrap: bool = True, time_limit: int = 1000):
        """
        Args:
            env: The environment to wrap.
            min: Minimum reward value.
            max: Maximum reward value.
        """
        super().__init__(env, unwrap=unwrap)
        self.time_limit = time_limit

    def init(self, key: Key) -> tuple[TimeLimitState, Timestep]:
        env_state, timestep = self.env.init(key)
        time_limit_state = TimeLimitState(
            env_state=env_state,
            time=jnp.array(0, dtype=jnp.int32),
        )
        return time_limit_state, timestep

    def reset(
        self, key: Key, env_state: TimeLimitState
    ) -> tuple[TimeLimitState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        time_limit_state = TimeLimitState(
            env_state=env_state,
            time=jnp.array(0, dtype=jnp.int32),
        )
        return time_limit_state, timestep

    def step(
        self,
        key: Key,
        env_state: TimeLimitState,
        action: PyTree,
    ) -> tuple[TimeLimitState, Timestep]:
        step_env_state, timestep = self.env.step(key, env_state.env_state, action)
        time_limit_state = TimeLimitState(
            env_state=step_env_state,
            time=env_state.time + 1,
        )
        reset_env_state, reset_timestep = self.env.reset(key, env_state.env_state)
        reset_time_limit_state = TimeLimitState(
            env_state=reset_env_state,
            time=jnp.array(0, dtype=jnp.int32),
        )
        reset_timestep = Timestep(
            obs=reset_timestep.obs,
            true_obs=timestep.obs,
            reward=timestep.reward,
            terminated=jnp.array(False, dtype=jnp.bool),
            truncated=jnp.array(True, dtype=jnp.bool),
            info=timestep.info,
        )
        time_limit_state, timestep = jax.lax.cond(
            jnp.logical_and(env_state.time + 1 >= self.time_limit, ~timestep.done),
            lambda: (reset_time_limit_state, reset_timestep),
            lambda: (time_limit_state, timestep),
        )
        return time_limit_state, timestep
