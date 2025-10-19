import pytest
from test_wrapper import TestWrapper

from gxm.wrappers import RecordEpisodeStatistics, Wrapper


class TestRecordEpisodeStatistics(TestWrapper):

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return RecordEpisodeStatistics(env)
