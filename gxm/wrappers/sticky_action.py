from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class StickyActionState(WrapperState):
    prev_action: Array


class StickyAction(Wrapper):
    """A wrapper that makes actions sticky with a given probability."""

    def __init__(self, env: Environment, stickiness: float = 0.25):
        self.env = env
        self.stickiness = stickiness

    def init(self, key: Array) -> tuple[StickyActionState, Timestep]:
        env_state, timestep = self.env.init(key)
        sticky_action_state = StickyActionState(
            env_state=env_state,
            prev_action=self.env.action_space.sample(key),
        )
        return sticky_action_state, timestep

    def reset(
        self, key: Array, env_state: StickyActionState
    ) -> tuple[StickyActionState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        sticky_action_state = StickyActionState(
            env_state=env_state,
            prev_action=self.env.action_space.sample(key),
        )
        return sticky_action_state, timestep

    def step(
        self,
        key: Array,
        env_state: StickyActionState,
        action: Array,
    ) -> tuple[StickyActionState, Timestep]:
        sticky_action = jnp.where(
            jax.random.uniform(key) < self.stickiness,
            env_state.prev_action,
            action,
        )
        env_state, timestep = self.env.step(key, env_state.env_state, sticky_action)
        sticky_action_state = StickyActionState(
            env_state=env_state,
            prev_action=action,
        )
        return sticky_action_state, timestep
