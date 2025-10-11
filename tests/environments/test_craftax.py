import jax
import pytest
from craftax.craftax_env import make_craftax_env_from_name
from test_environment import TestEnvironment

import gxm


class TestCraftax(TestEnvironment):
    @pytest.fixture(
        params=[
            "Craftax-Symbolic-v1",
            "Craftax-Pixels-v1",
        ]
    )
    def env(self, request):
        return gxm.make("Craftax/" + request.param)

    @pytest.fixture(
        params=[
            "Craftax-Symbolic-v1",
            "Craftax-Pixels-v1",
        ]
    )
    def id(self, request):
        return request.param

    def test_equality(self, id):
        env_gxm = gxm.make("Craftax/" + id)
        env_craftax = make_craftax_env_from_name(id, auto_reset=True)
        env_params_craftax = env_craftax.default_params

        key = jax.random.key(0)
        env_state, timestep = env_gxm.init(key)
        obs, state = env_craftax.reset(key, env_params_craftax)

        for _ in range(1000):
            action = env_gxm.action_space.sample(key)
            env_state, timestep = env_gxm.step(key, env_state, action)
            obs, state, reward, done, info = env_craftax.step(
                key, state, action, env_params_craftax
            )
            assert jax.numpy.allclose(timestep.obs, obs)
            assert jax.numpy.allclose(timestep.reward, reward)
            assert timestep.done == done
