from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class StickyActionState(WrapperState):
    prev_action: PyTree


class StickyAction(Wrapper):
    """A wrapper that makes actions sticky with a given probability."""

    def __init__(self, env: Environment, unwrap: bool = True, stickiness: float = 0.25):
        super().__init__(env, unwrap=unwrap)
        self.stickiness = stickiness

    def init(self, key: Key) -> tuple[StickyActionState, Timestep]:
        env_state, timestep = self.env.init(key)
        sticky_action_state = StickyActionState(
            env_state=env_state,
            prev_action=self.env.action_space.sample(key),
        )
        return sticky_action_state, timestep

    def reset(
        self, key: Key, env_state: StickyActionState
    ) -> tuple[StickyActionState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        sticky_action_state = StickyActionState(
            env_state=env_state,
            prev_action=self.env.action_space.sample(key),
        )
        return sticky_action_state, timestep

    def step(
        self,
        key: Key,
        env_state: StickyActionState,
        action: PyTree,
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
