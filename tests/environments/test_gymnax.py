import gymnax
import jax
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
    def id(self, request):
        return request.param

    def test_equality(self, id):
        env_gxm = gxm.make("Gymnax/" + id)
        env_gymnax, env_params_gymnax = gymnax.make(id)

        key = jax.random.key(0)
        env_state, timestep = env_gxm.init(key)
        obs, state = env_gymnax.reset(key)

        for _ in range(1000):
            action = env_gxm.action_space.sample(key)
            env_state, timestep = env_gxm.step(key, env_state, action)
            obs, state, reward, done, info = env_gymnax.step(
                key, state, action, env_params_gymnax
            )
            assert jax.numpy.allclose(timestep.obs, obs)
            assert jax.numpy.allclose(timestep.reward, reward)
            assert timestep.done == done
