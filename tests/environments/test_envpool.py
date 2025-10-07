import envpool
import pytest
from test_environment import TestEnvironment

import gxm


class TestGymnax(TestEnvironment):
    @pytest.fixture(params=envpool.list_all_envs()[:10])
    def env(self, request):
        return gxm.make("Envpool/" + request.param)
