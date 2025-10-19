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
