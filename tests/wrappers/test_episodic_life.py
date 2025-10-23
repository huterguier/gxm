import ale_py
import pytest
from test_wrapper import TestWrapper

import gxm
from gxm.core import Environment
from gxm.wrappers import EpisodicLife, Wrapper


class TestEpisodicLife(TestWrapper):

    @pytest.fixture(
        params=[
            "Gymnasium/ALE/Breakout-v5",
            "Gymnasium/ALE/SpaceInvaders-v5",
        ]
    )
    def env(self, request) -> Environment:
        return gxm.make(request.param)

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return EpisodicLife(env)
