import pytest
from test_environment import TestEnvironment

import gxm


class TestGymnax(TestEnvironment):
    @pytest.fixture(
        params=[
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1",
            "MountainCarContinuous-v0",
        ]
    )
    def env(self, request):
        return gxm.make("Gymnasium/" + request.param)
