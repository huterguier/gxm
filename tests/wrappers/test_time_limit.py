import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import TimeLimit, Wrapper


class TestStickyAction(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return TimeLimit(env, time_limit=100)
