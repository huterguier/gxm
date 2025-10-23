import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import FlattenObservation, Wrapper


class TestDiscretize(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return FlattenObservation(env)
