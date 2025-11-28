import pytest
from test_environment import TestEnvironment

import gxm


class TestGymnax(TestEnvironment):
    @pytest.fixture(
        params=[
            "XLand-MiniGrid-R1-9x9",
        ]
    )
    def env(self, request):
        return gxm.make("XMiniGrid/" + request.param)
