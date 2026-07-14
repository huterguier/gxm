import pytest

pytest.importorskip("xminigrid")
import jax
import jax.numpy as jnp
from test_environment import TestEnvironment

import gxm
from gxm.adapters.xminigrid import XMiniGridAdapter
from gxm.wrappers import AutoReset


class TestXMiniGrid(TestEnvironment):
    @pytest.fixture(
        params=[
            "XLand-MiniGrid-R1-9x9",
        ]
    )
    def env(self, request):
        return gxm.make("XMiniGrid/" + request.param)

    def test_make_autoreset_default(self):
        env = gxm.make("XMiniGrid/XLand-MiniGrid-R1-9x9")
        assert env.has_wrapper(AutoReset)
        assert env.unwrapped is env
        raw = gxm.make("XMiniGrid/XLand-MiniGrid-R1-9x9", autoreset=False)
        assert isinstance(raw, XMiniGridAdapter)

    def test_truncation_is_labeled(self):
        """Turning in place never reaches a goal, so max_steps=5 guarantees
        the episode ends by truncation, never termination."""
        key = jax.random.key(0)
        env = gxm.make("XMiniGrid/XLand-MiniGrid-R1-9x9")
        adapter = env.get_wrapper(XMiniGridAdapter)
        adapter.env_params = adapter.env_params.replace(max_steps=5)
        env_state, timestep = env.init(key)
        for _ in range(5):
            key, key_step = jax.random.split(key)
            env_state, timestep = env.step(key_step, env_state, jnp.int32(1))

        assert bool(timestep.truncated)
        assert not bool(timestep.terminated)
