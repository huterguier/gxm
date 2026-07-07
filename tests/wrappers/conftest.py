from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from gxm.core import Model, ModelState, Step
from gxm.spaces import Box, Discrete
from gxm.typing import Key, PyTree


@jax.tree_util.register_dataclass
@dataclass
class DummyModelState(ModelState):
    count: jnp.ndarray


class DummyModel(Model[DummyModelState, Step]):
    """A minimal bare Model (not an Environment), for testing dynamics-only wrappers."""

    id = "DummyModel"
    action_space = Discrete(2)
    observation_space = Box(low=0.0, high=1.0, shape=(4,))

    def init(self, key: Key) -> tuple[DummyModelState, Step]:
        return self.reset(key, DummyModelState(count=jnp.int32(0)))

    def reset(self, key: Key, state: DummyModelState) -> tuple[DummyModelState, Step]:
        obs = self.observation_space.sample(key)
        return DummyModelState(count=jnp.int32(0)), Step(
            next_obs=obs, action=self.action_space.sample(key), info={}
        )

    def step(
        self, key: Key, state: DummyModelState, action: PyTree
    ) -> tuple[DummyModelState, Step]:
        obs = self.observation_space.sample(key)
        state = DummyModelState(count=state.count + 1)
        return state, Step(next_obs=obs, action=action, info={"count": state.count})


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel()
