import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import AutoReset, Wrapper


class TestAutoReset(TestWrapper):
    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return AutoReset(env)
