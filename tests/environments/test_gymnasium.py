import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("ale_py")
import gymnasium
import jax
import numpy as np
from ale_py import ALEInterface
from test_environment import TestEnvironment

import gxm
from gxm.adapters.gymnasium import GymnasiumState

ale = ALEInterface()


class TestGymnasium(TestEnvironment):
    @pytest.fixture(
        params=[
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1",
            "MountainCarContinuous-v0",
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

    def test_vmap(self, env):
        key = jax.random.key(0)
        keyss = jax.random.split(key, (10, 10))

        def rollout(key):
            env_state, _ = env.init(key)

            def step(env_state, key):
                action = env.action_space.sample(key)
                return env.step(key, env_state, action)

            keys = jax.random.split(key, 100)
            env_state, timesteps = jax.lax.scan(step, env_state, keys)
            return timesteps

        _ = jax.vmap(jax.vmap(rollout))(keyss)

    def test_equality(self, id):
        env_gxm = gxm.make("Gymnasium/" + id)
        env_gymnasium = gymnasium.make_vec(id)

        key = jax.random.key(0)
        env_state, timestep = env_gxm.init(key)

        obs, _ = env_gymnasium.reset(seed=gxm.gymnasium.seed_from_key(key))
        assert jax.numpy.allclose(timestep.next_obs, obs[0])

        for _ in range(100):
            action = env_gxm.action_space.sample(key)
            env_state, timestep = env_gxm.step(key, env_state, action)
            obs, reward, terminated, truncated, _ = env_gymnasium.step(
                np.array([action])
            )
            assert jax.numpy.allclose(timestep.next_obs, obs)
            assert jax.numpy.allclose(timestep.reward, reward)
            assert timestep.terminated == terminated
            assert timestep.truncated == truncated

    def test_spliced_state_raises(self):
        env = gxm.make("Gymnasium/CartPole-v1")
        a, _ = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 4))
        b, _ = jax.vmap(env.init)(jax.random.split(jax.random.key(1), 4))
        state = GymnasiumState(
            env_id=jax.numpy.concatenate([a.env_id[:2], b.env_id[:2]])
        )
        action = jax.numpy.zeros((4,), jax.numpy.int32)
        with pytest.raises(Exception, match="mixes env ids"):
            jax.vmap(env.step, in_axes=(None, 0, 0))(jax.random.key(0), state, action)

    def test_sliced_state_raises(self):
        env = gxm.make("Gymnasium/CartPole-v1")
        state, _ = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 4))
        state = GymnasiumState(env_id=state.env_id[:2])
        with pytest.raises(Exception, match="batch size 2"):
            jax.vmap(env.reset, in_axes=(None, 0))(jax.random.key(0), state)

    def test_close(self):
        env = gxm.make("Gymnasium/CartPole-v1")
        state, _ = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 4))
        assert len(env._envs) == 1
        env.close()
        assert len(env._envs) == 0
        action = jax.numpy.zeros((4,), jax.numpy.int32)
        with pytest.raises(Exception, match="not live"):
            jax.vmap(env.step, in_axes=(None, 0, 0))(jax.random.key(0), state, action)

    def test_env_ids_are_not_recycled(self):
        env = gxm.make("Gymnasium/CartPole-v1")
        before, _ = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 4))
        env.close()
        after, _ = jax.vmap(env.init)(jax.random.split(jax.random.key(1), 4))
        assert int(after.env_id[0]) != int(before.env_id[0])

    def test_reset_uses_key(self):
        env = gxm.make("Gymnasium/CartPole-v1")
        key = jax.random.key(0)
        state, initial = env.init(key)
        _, same = env.reset(key, state)
        _, other = env.reset(jax.random.key(1), state)
        assert jax.numpy.allclose(initial.next_obs, same.next_obs)
        assert not jax.numpy.allclose(same.next_obs, other.next_obs)
