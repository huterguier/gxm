import gymnax
import jax
import jax.numpy as jnp

import gxm


class TestTerminalWrappers:
    def test_gymnax_to_gxm(self):
        gymnax_env, params = gymnax.make("CartPole-v1")
        gxm_env = gxm.gymnax.wrap(gymnax_env, params)

        key = jax.random.PRNGKey(0)
        state, timestep = gxm_env.init(key)
        assert timestep.next_obs.shape == (4,)

        action = gxm_env.action_space.sample(key)
        next_state, next_timestep = gxm_env.step(key, state, action)
        assert next_timestep.next_obs.shape == (4,)

    def test_gxm_to_gymnax(self):
        gxm_env = gxm.make("Gymnax/CartPole-v1")
        gymnax_wrapper = gxm.gymnax.unwrap(gxm_env)

        key = jax.random.PRNGKey(0)
        obs, state = gymnax_wrapper.reset(key)
        assert obs.shape == (4,)

        action = gxm_env.action_space.sample(key)
        obs, state, reward, done, info = gymnax_wrapper.step(key, state, action)
        assert obs.shape == (4,)
        assert isinstance(reward, jax.Array)
        assert isinstance(done, jax.Array)

    def test_round_trip(self):
        gymnax_env, params = gymnax.make("CartPole-v1")
        gxm_env = gxm.gymnax.wrap(gymnax_env, params)
        restored_gymnax = gxm.gymnax.unwrap(gxm_env)

        key = jax.random.PRNGKey(0)
        obs1, state1 = gymnax_env.reset(key, params)
        obs2, state2 = restored_gymnax.reset(key)
        assert jnp.allclose(obs1, obs2)

        action = 1
        obs1, state1, r1, d1, _ = gymnax_env.step(key, state1, action, params)
        obs2, state2, r2, d2, _ = restored_gymnax.step(key, state2, action)
        assert jnp.allclose(obs1, obs2)
        assert jnp.allclose(r1, r2)
        assert d1 == d2
