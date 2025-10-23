import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import StackObservations, Wrapper


class TestStackObservation(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return StackObservations(env, num_stack=4)
