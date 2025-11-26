from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class StepCounterState(WrapperState):
    n_steps: Array


class StepCounter(Wrapper):
    """A wrapper that counts the number of steps taken in the environment."""

    def __init__(self, env: Environment, unwrap: bool = True):
        super().__init__(env, unwrap=unwrap)

    def init(self, key: Key) -> tuple[StepCounterState, Timestep]:
        env_state, timestep = self.env.init(key)
        step_counter_state = StepCounterState(
            env_state=env_state,
            n_steps=jnp.int32(0),
        )
        timestep.info["n_steps"] = step_counter_state.n_steps
        return step_counter_state, timestep

    def reset(
        self, key: Key, env_state: StepCounterState
    ) -> tuple[StepCounterState, Timestep]:
        step_counter_state = env_state
        env_state, timestep = self.env.reset(key, step_counter_state.env_state)
        step_counter_state = StepCounterState(
            env_state=env_state,
            n_steps=step_counter_state.n_steps,
        )
        timestep.info["n_steps"] = step_counter_state.n_steps
        return step_counter_state, timestep

    def step(
        self,
        key: Key,
        env_state: StepCounterState,
        action: PyTree,
    ) -> tuple[StepCounterState, Timestep]:
        step_counter_state = env_state
        env_state, timestep = self.env.step(key, env_state.env_state, action)
        step_counter_state = StepCounterState(
            env_state=env_state,
            n_steps=step_counter_state.n_steps + 1,
        )
        timestep.info["n_steps"] = step_counter_state.n_steps
        return step_counter_state, timestep
