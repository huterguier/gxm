import jax
import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import Evaluate, Wrapper


class TestEvaluate(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return Evaluate(env)

    def test_evaluate(self, wrapper: Wrapper):
        key = jax.random.key(0)
        env_state, timestep = wrapper.init(key)

        def step(carry, _):
            env_state = carry
            action = wrapper.action_space.sample(key)
            env_state, _ = wrapper.step(key, env_state, action)
            return env_state, _

        env_state, _ = jax.lax.scan(step, env_state, length=1000)
        assert timestep.reward is not None
