import envpool
import jax
import numpy as np
import pytest
from test_environment import TestEnvironment

import gxm


class TestEnvpool(TestEnvironment):
    @pytest.fixture(params=envpool.list_all_envs()[:10])
    def env(self, request):
        return gxm.make("Envpool/" + request.param)

    @pytest.fixture(
        params=[
            "CartPole-v1",
            "Breakout-v5",
        ]
    )
    def id(self, request):
        return request.param

    def test_equality(self, id):
        env_gxm = gxm.make("Envpool/" + id)
        env_gymnasium = envpool.make(id, env_type="gym")

        key = jax.random.key(0)
        env_state, timestep = env_gxm.init(key)

        obs = env_gymnasium.reset()
        assert jax.numpy.allclose(timestep.obs, obs[0])

        for _ in range(100):
            action = env_gxm.action_space.sample(key)
            env_state, timestep = env_gxm.step(key, env_state, action)
            obs, reward, done, info = env_gymnasium.step(np.array([action]))
            # print number of elements in info dict
            print(len(info))
            assert jax.numpy.allclose(timestep.obs, obs)
            assert jax.numpy.allclose(timestep.reward, reward)
            assert timestep.done == done
