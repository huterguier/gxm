import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import EpisodeCounter, Wrapper


class TestEpisodeCounter(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return EpisodeCounter(env)
