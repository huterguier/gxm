from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from gxm.core import Dynamics, DynamicsState, Step
from gxm.spaces import Box, Discrete
from gxm.typing import Key, PyTree


@jax.tree_util.register_dataclass
@dataclass
class DummyDynamicsState(DynamicsState):
    count: jnp.ndarray


class DummyDynamics(Dynamics[DummyDynamicsState, Step]):
    """Minimal bare Dynamics (not an Environment), for testing dynamics-only wrappers."""

    id = "DummyDynamics"
    action_space = Discrete(2)
    observation_space = Box(low=0.0, high=1.0, shape=(4,))

    def init(self, key: Key) -> tuple[DummyDynamicsState, Step]:
        return self.reset(key, DummyDynamicsState(count=jnp.int32(0)))

    def reset(
        self, key: Key, state: DummyDynamicsState
    ) -> tuple[DummyDynamicsState, Step]:
        obs = self.observation_space.sample(key)
        return DummyDynamicsState(count=jnp.int32(0)), Step(
            next_obs=obs, action=self.action_space.sample(key), info={}
        )

    def step(
        self, key: Key, state: DummyDynamicsState, action: PyTree
    ) -> tuple[DummyDynamicsState, Step]:
        obs = self.observation_space.sample(key)
        state = DummyDynamicsState(count=state.count + 1)
        return state, Step(next_obs=obs, action=action, info={"count": state.count})


@pytest.fixture
def dummy_dynamics() -> DummyDynamics:
    return DummyDynamics()
