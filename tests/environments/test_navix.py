import pytest
from test_environment import TestEnvironment

import gxm


class TestGymnax(TestEnvironment):
    @pytest.fixture(
        params=[
            "Navix-Empty-5x5-v0",
        ]
    )
    def env(self, request):
        return gxm.make("Navix/" + request.param)
