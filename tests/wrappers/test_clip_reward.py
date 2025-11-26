import jax
import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import ClipReward, Wrapper


class TestDiscretize(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return ClipReward(env)

    def test_reward_clipping(self, wrapper):
        key = jax.random.key(0)
        wrapper_state, _ = wrapper.init(key)

        for _ in range(100):
            key, key_step = jax.random.split(key)
            action = wrapper.action_space.sample(key)
            wrapper_state, timestep = wrapper.step(key_step, wrapper_state, action)
            assert -1.0 <= timestep.reward <= 1.0
