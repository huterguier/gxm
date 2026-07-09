from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from gxm.core import Dynamics, TStep
from gxm.typing import Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class StickyActionState(WrapperState):
    prev_action: PyTree


class StickyAction(Wrapper[StickyActionState, TStep]):
    """A wrapper that makes actions sticky with a given probability."""

    def __init__(
        self,
        wrapped: Dynamics[Any, TStep],
        unwrap: bool = True,
        stickiness: float = 0.25,
    ):
        super().__init__(wrapped, unwrap=unwrap)
        self.stickiness = stickiness

    def init(self, key: Key) -> tuple[StickyActionState, TStep]:
        wrapped_state, step = self.wrapped.init(key)
        sticky_action_state = StickyActionState(
            wrapped_state=wrapped_state,
            prev_action=self.wrapped.action_space.sample(key),
        )
        return sticky_action_state, step

    def reset(
        self, key: Key, state: StickyActionState
    ) -> tuple[StickyActionState, TStep]:
        wrapped_state, step = self.wrapped.reset(key, state)
        sticky_action_state = StickyActionState(
            wrapped_state=wrapped_state,
            prev_action=self.wrapped.action_space.sample(key),
        )
        return sticky_action_state, step

    def step(
        self,
        key: Key,
        state: StickyActionState,
        action: PyTree,
    ) -> tuple[StickyActionState, TStep]:
        sticky_action = jnp.where(
            jax.random.uniform(key) < self.stickiness,
            state.prev_action,
            action,
        )
        wrapped_state, step = self.wrapped.step(key, state.wrapped_state, sticky_action)
        sticky_action_state = StickyActionState(
            wrapped_state=wrapped_state,
            prev_action=sticky_action,
        )
        return sticky_action_state, step
