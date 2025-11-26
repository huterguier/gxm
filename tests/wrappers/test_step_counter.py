import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import StepCounter, Wrapper


class TestStepCounter(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return StepCounter(env)
