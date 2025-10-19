import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import ClipReward, Wrapper


class TestClipReward(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return ClipReward(env)
