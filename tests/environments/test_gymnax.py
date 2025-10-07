import pytest
from test_environment import TestEnvironment

import gxm


class TestGymnax(TestEnvironment):
    @pytest.fixture(
        params=[
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Reacher-misc",
        ]
    )
    def env(self, request):
        return gxm.make("Gymnax/" + request.param)
