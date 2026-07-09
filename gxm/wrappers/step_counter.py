import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from gxm.core import Dynamics, TStep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class StepCounterState(WrapperState):
    n_steps: Array


class StepCounter(Wrapper[StepCounterState, TStep]):
    """A wrapper that counts the number of steps taken in the environment."""

    def __init__(self, wrapped: Dynamics[Any, TStep], unwrap: bool = True):
        super().__init__(wrapped, unwrap=unwrap)

    def init(self, key: Key) -> tuple[StepCounterState, TStep]:
        wrapped_state, step_output = self.wrapped.init(key)
        step_counter_state = StepCounterState(
            wrapped_state=wrapped_state,
            n_steps=jnp.int32(0),
        )
        step_output = dataclasses.replace(
            step_output, info=step_output.info | {"n_steps": step_counter_state.n_steps}
        )
        return step_counter_state, step_output

    def reset(
        self, key: Key, state: StepCounterState
    ) -> tuple[StepCounterState, TStep]:
        step_counter_state = state
        wrapped_state, step_output = self.wrapped.reset(
            key, step_counter_state.wrapped_state
        )
        step_counter_state = StepCounterState(
            wrapped_state=wrapped_state,
            n_steps=step_counter_state.n_steps,
        )
        step_output = dataclasses.replace(
            step_output, info=step_output.info | {"n_steps": step_counter_state.n_steps}
        )
        return step_counter_state, step_output

    def step(
        self,
        key: Key,
        state: StepCounterState,
        action: PyTree,
    ) -> tuple[StepCounterState, TStep]:
        step_counter_state = state
        wrapped_state, step_output = self.wrapped.step(key, state.wrapped_state, action)
        step_counter_state = StepCounterState(
            wrapped_state=wrapped_state,
            n_steps=step_counter_state.n_steps + 1,
        )
        step_output = dataclasses.replace(
            step_output, info=step_output.info | {"n_steps": step_counter_state.n_steps}
        )
        return step_counter_state, step_output
