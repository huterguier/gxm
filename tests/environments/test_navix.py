import pytest

pytest.importorskip("navix")
import jax
import jax.numpy as jnp
from test_environment import TestEnvironment

import gxm
from gxm.adapters.navix import NavixAdapter
from gxm.wrappers import AutoReset


class TestNavix(TestEnvironment):
    @pytest.fixture(
        params=[
            "Navix-Empty-5x5-v0",
        ]
    )
    def env(self, request):
        return gxm.make("Navix/" + request.param)

    def test_make_autoreset_default(self):
        env = gxm.make("Navix/Navix-Empty-5x5-v0")
        assert env.has_wrapper(AutoReset)
        assert env.unwrapped is env
        raw = gxm.make("Navix/Navix-Empty-5x5-v0", autoreset=False)
        assert isinstance(raw, NavixAdapter)

    def test_truncation_is_labeled(self):
        """Rotating in place never reaches the goal, so max_steps=5 guarantees
        the episode ends by truncation, never termination."""
        key = jax.random.key(0)
        env = gxm.make("Navix/Navix-Empty-5x5-v0", max_steps=5)
        env_state, timestep = env.init(key)
        for _ in range(5):
            key, key_step = jax.random.split(key)
            env_state, timestep = env.step(key_step, env_state, jnp.int32(0))

        assert bool(timestep.truncated)
        assert not bool(timestep.terminated)
