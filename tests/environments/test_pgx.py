import pytest

pytest.importorskip("pgx")
from test_environment import TestEnvironment

import gxm
from gxm.adapters.pgx import PgxAdapter
from gxm.wrappers import AutoReset


class TestPgx(TestEnvironment):
    @pytest.fixture(
        params=[
            "2048",
            "minatar-breakout",
        ]
    )
    def env(self, request):
        return gxm.make("Pgx/" + request.param)

    def test_make_autoreset_default(self):
        env = gxm.make("Pgx/2048")
        assert env.has_wrapper(AutoReset)
        assert env.unwrapped is env
        raw = gxm.make("Pgx/2048", autoreset=False)
        assert isinstance(raw, PgxAdapter)
