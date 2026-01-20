import brax.envs
import jax
import pytest
from test_environment import TestEnvironment

import gxm


class TestBrax(TestEnvironment):
    @pytest.fixture(
        params=[
            "ant",
            "halfcheetah",
        ]
    )
    def env(self, request):
        return gxm.make("Brax/" + request.param)

    @pytest.fixture(
        params=[
            "ant",
            "halfcheetah",
        ]
    )
    def id(self, request):
        return request.param

    def test_equality(self, id):
        env_gxm = gxm.make("Brax/" + id)
        env_brax = brax.envs.create(id)

        key = jax.random.key(0)
        env_state, timestep = env_gxm.init(key)
        state_brax = env_brax.reset(key)

        assert jax.numpy.allclose(timestep.obs, state_brax.obs)

        for _ in range(10):
            key, subkey = jax.random.split(key)
            action = env_gxm.action_space.sample(subkey)
            
            env_state, timestep = env_gxm.step(subkey, env_state, action)
            state_brax = env_brax.step(state_brax, action)
            
            assert jax.numpy.allclose(timestep.obs, state_brax.obs)
            assert jax.numpy.allclose(timestep.reward, state_brax.reward)
            assert timestep.terminated == (state_brax.done > 0.5)
