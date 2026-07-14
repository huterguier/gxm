import pytest

pytest.importorskip("brax")
import brax.envs
import jax
from test_environment import TestEnvironment

import gxm


class TestBrax(TestEnvironment):
    # Class-scoped: brax env creation is expensive, and repeatedly creating
    # envs + XLA CPU executables in one process has been observed to segfault
    # in brax's native code. gxm envs are stateless, so reuse is safe.
    @pytest.fixture(
        scope="class",
        params=[
            "ant",
            "halfcheetah",
        ],
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
        env_brax = brax.envs.create(id, auto_reset=False)

        key = jax.random.key(0)
        env_state, timestep = env_gxm.init(key)
        state_brax = env_brax.reset(key)

        assert jax.numpy.allclose(timestep.next_obs, state_brax.obs)

        for _ in range(10):
            key, subkey = jax.random.split(key)
            action = env_gxm.action_space.sample(subkey)

            env_state, timestep = env_gxm.step(subkey, env_state, action)
            state_brax = env_brax.step(state_brax, action)

            assert jax.numpy.allclose(timestep.reward, state_brax.reward)
            assert timestep.done == (state_brax.done > 0.5)
            if bool(timestep.done):
                # true_next_obs preserves the terminal observation the
                # non-resetting reference env still shows.
                assert jax.numpy.allclose(timestep.true_next_obs, state_brax.obs)
            else:
                assert jax.numpy.allclose(timestep.next_obs, state_brax.obs)
            # Re-sync so both sides step from identical states even after a
            # done (gxm auto-resets, the reference does not).
            state_brax = env_state.brax_state

    def test_truncation_is_labeled(self):
        """A healthy ant cannot terminate in 5 steps, so episode_length=5
        guarantees the episode ends by truncation."""
        key = jax.random.key(0)
        env = gxm.make("Brax/ant", episode_length=5)
        env_state, timestep = env.init(key)
        for _ in range(5):
            key, key_step = jax.random.split(key)
            action = env.action_space.sample(key_step)
            env_state, timestep = env.step(key_step, env_state, action)

        assert bool(timestep.truncated)
        assert not bool(timestep.terminated)
        # next_obs is the fresh auto-reset world, true_next_obs the old one.
        assert not jax.numpy.allclose(timestep.true_next_obs, timestep.next_obs)
