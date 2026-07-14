import pytest

pytest.importorskip("craftax.craftax_env")
import jax
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
        env_craftax = make_craftax_env_from_name(id, auto_reset=False)
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
            assert jax.numpy.allclose(timestep.reward, reward)
            assert timestep.done == done
            if bool(timestep.done):
                # true_next_obs preserves the terminal observation the
                # non-resetting reference env still shows.
                assert jax.numpy.allclose(timestep.true_next_obs, obs)
            else:
                assert jax.numpy.allclose(timestep.next_obs, obs)
            # Re-sync so both sides step from identical states even after a
            # done (gxm auto-resets, the reference does not).
            state = env_state.craftax_state

    def test_truncation_is_labeled(self):
        """Random actions do not kill the agent within 5 steps of spawning, so
        max_timesteps=5 guarantees the episode ends by truncation."""
        from gxm.adapters.craftax import CraftaxAdapter

        key = jax.random.key(0)
        env = gxm.make("Craftax/Craftax-Symbolic-v1")
        adapter = env.get_wrapper(CraftaxAdapter)
        adapter.env_params = adapter.env_params.replace(max_timesteps=5)
        env_state, timestep = env.init(key)
        for _ in range(5):
            key, key_step = jax.random.split(key)
            action = env.action_space.sample(key_step)
            env_state, timestep = env.step(key_step, env_state, action)

        assert bool(timestep.truncated)
        assert not bool(timestep.terminated)
        # next_obs is the fresh auto-reset world, true_next_obs the old one.
        assert not jax.numpy.allclose(timestep.true_next_obs, timestep.next_obs)
