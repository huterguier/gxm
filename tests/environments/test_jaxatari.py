import pytest

pytest.importorskip("jaxatari")
from test_environment import TestEnvironment

import gxm


class TestJAXAtari(TestEnvironment):
    # Class-scoped: creating a jaxatari environment compiles the full pixel
    # pipeline eagerly (info-structure probe), which is expensive on CPU.
    # gxm envs are stateless, so reuse across tests is safe.
    @pytest.fixture(
        scope="class",
        params=[
            "pong",
        ],
    )
    def env(self, request):
        return gxm.make("JAXAtari/" + request.param)

    def test_make_autoreset_false_raises(self):
        """jaxatari auto-resets internally with no opt-out, so a
        non-resetting environment cannot be constructed."""
        with pytest.raises(ValueError):
            gxm.make("JAXAtari/pong", autoreset=False)
