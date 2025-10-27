import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import StickyAction, Wrapper


class TestDiscretize(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return StickyAction(env, stickiness=0.1)
