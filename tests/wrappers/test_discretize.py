import jax
import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import Discretize, Wrapper


class TestDiscretize(TestWrapper):
    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        key = jax.random.key(0)
        actions = env.action_space.sample(key, (5,))
        return Discretize(env, actions)

    def test_wraps_bare_dynamics(self, dummy_dynamics):
        key = jax.random.key(0)
        actions = dummy_dynamics.action_space.sample(key, (5,))
        wrapper = Discretize(dummy_dynamics, actions)
        state, step = wrapper.init(key)
        assert not hasattr(step, "reward")
        action = wrapper.action_space.sample(key)
        state, step = wrapper.step(key, state, action)
        assert not hasattr(step, "reward")
