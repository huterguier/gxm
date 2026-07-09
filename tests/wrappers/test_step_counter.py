import jax
import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import StepCounter, Wrapper


class TestStepCounter(TestWrapper):
    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return StepCounter(env)

    def test_wraps_bare_dynamics(self, dummy_dynamics):
        key = jax.random.key(0)
        wrapper = StepCounter(dummy_dynamics)
        state, step = wrapper.init(key)
        assert not hasattr(step, "reward")
        action = wrapper.action_space.sample(key)
        state, step = wrapper.step(key, state, action)
        assert not hasattr(step, "reward")
