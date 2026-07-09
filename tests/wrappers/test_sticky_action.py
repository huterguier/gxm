import jax
import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import StickyAction, Wrapper


class TestStickyAction(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return StickyAction(env, stickiness=0.1)

    def test_wraps_bare_dynamics(self, dummy_dynamics):
        key = jax.random.key(0)
        wrapper = StickyAction(dummy_dynamics, stickiness=0.1)
        state, step = wrapper.init(key)
        assert not hasattr(step, "reward")
        action = wrapper.action_space.sample(key)
        state, step = wrapper.step(key, state, action)
        assert not hasattr(step, "reward")
