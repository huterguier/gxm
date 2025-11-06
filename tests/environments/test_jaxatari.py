import pytest
from test_environment import TestEnvironment

import gxm


class TestJAXAtari(TestEnvironment):
    @pytest.fixture(
        params=[
            "pong",
            "breakout",
        ]
    )
    def env(self, request):
        return gxm.make("JAXAtari/" + request.param)
