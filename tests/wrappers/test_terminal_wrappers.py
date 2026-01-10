import gymnax
import jax
import jax.numpy as jnp
import pytest

import gxm
from gxm.wrappers.terminal import GxmToGymnax, GymnaxToGxm


class TestTerminalWrappers:
    def test_gymnax_to_gxm(self):
        env_id = "CartPole-v1"
        gymnax_env, params = gymnax.make(env_id)

        gxm_env = GymnaxToGxm(gymnax_env, params)

        # Test basic GXM API
        key = jax.random.PRNGKey(0)
        state, timestep = gxm_env.init(key)
        assert timestep.obs.shape == (4,)

        action = gxm_env.action_space.sample(key)
        next_state, next_timestep = gxm_env.step(key, state, action)

        assert next_timestep.obs.shape == (4,)

    def test_gxm_to_gymnax(self):
        # Create a GXM env (e.g. wrapper around gymnax via gxm.make is fine, or the one we just made)
        gxm_env = gxm.make("Gymnax/CartPole-v1")

        gymnax_wrapper = GxmToGymnax(gxm_env)

        # Test basic Gymnax API
        key = jax.random.PRNGKey(0)
        obs, state = gymnax_wrapper.reset(key)
        assert obs.shape == (4,)

        # Use gxm space to sample to ensure compatibility
        action = gxm_env.action_space.sample(key)

        obs, state, reward, done, info = gymnax_wrapper.step(key, state, action)

        assert obs.shape == (4,)
        assert isinstance(reward, jax.Array)
        assert isinstance(done, jax.Array)

    def test_round_trip(self):
        env_id = "CartPole-v1"
        gymnax_env, params = gymnax.make(env_id)

        # Gymnax -> Gxm -> Gymnax
        gxm_env = GymnaxToGxm(gymnax_env, params)
        restored_gymnax = GxmToGymnax(gxm_env)

        key = jax.random.PRNGKey(0)
        obs1, state1 = gymnax_env.reset(key, params)
        obs2, state2 = restored_gymnax.reset(key)  # Params ignored by wrapper

        assert jnp.allclose(obs1, obs2)

        action = 1
        obs1, state1, r1, d1, _ = gymnax_env.step(key, state1, action, params)
        obs2, state2, r2, d2, _ = restored_gymnax.step(key, state2, action)

        assert jnp.allclose(obs1, obs2)
        assert jnp.allclose(r1, r2)
        # assert d1 == d2 # Done might differ slightly if one uses truncated=False explicitly?
        # Gymnax env.step returns done. GymnaxToGxm sets terminated=done, truncated=False.
        # GxmToGymnax returns done = terminated | truncated.
        # So it should be equal.
        assert d1 == d2
