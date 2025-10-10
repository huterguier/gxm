import gymnasium
import jax
import numpy as np
import pytest
from ale_py import ALEInterface
from test_environment import TestEnvironment

import gxm

ale = ALEInterface()


class TestGymnax(TestEnvironment):
    @pytest.fixture(
        params=[
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1",
            # "MountainCarContinuous-v0",
        ]
    )
    def env(self, request):
        return gxm.make("Gymnasium/" + request.param)

    @pytest.fixture(
        params=[
            "CartPole-v1",
            "ALE/Breakout-v5",
        ]
    )
    def id(self, request):
        return request.param

    def test_equality(self, id):
        env_gxm = gxm.make("Gymnasium/" + id)
        env_gymnasium = gymnasium.make_vec(id)

        key = jax.random.key(0)
        env_state, timestep = env_gxm.init(key)

        obs, _ = env_gymnasium.reset(seed=0)
        assert jax.numpy.allclose(timestep.obs, obs[0])

        for _ in range(100):
            action = env_gxm.action_space.sample(key)
            env_state, timestep = env_gxm.step(key, env_state, action)
            obs, reward, terminated, truncated, _ = env_gymnasium.step(
                np.array([action])
            )
            assert jax.numpy.allclose(timestep.obs, obs)
            assert jax.numpy.allclose(timestep.reward, reward)
            assert timestep.terminated == terminated
            assert timestep.truncated == truncated
