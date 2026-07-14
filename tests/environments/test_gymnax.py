import pytest

pytest.importorskip("gymnax")
import gymnax
import jax
import jax.numpy as jnp
from test_environment import TestEnvironment

import gxm
from gxm.adapters.gymnax import GymnaxAdapter
from gxm.wrappers import AutoReset


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
            assert jax.numpy.allclose(timestep.next_obs, obs, atol=1e-6)
            assert jax.numpy.allclose(timestep.reward, reward)
            assert timestep.done == done
            # Re-sync the reference env to gxm's state so single-step outputs
            # are compared from identical inputs. The two sides now compile
            # step_env in different jit contexts, and the resulting ulp-level
            # rounding differences compound chaotically (Acrobot, Reacher)
            # over long horizons.
            state = env_state.gymnax_state

    def test_make_autoreset_default(self):
        env = gxm.make("Gymnax/CartPole-v1")
        assert env.has_wrapper(AutoReset)
        # The auto-reset layer is sealed: unwrapped does not peel it off, so
        # nobody silently ends up with a non-resetting environment.
        assert env.unwrapped is env
        assert isinstance(env.get_wrapper(AutoReset), AutoReset)
        raw = gxm.make("Gymnax/CartPole-v1", autoreset=False)
        assert isinstance(raw, GymnaxAdapter)

    def test_truncation_is_labeled(self):
        """MountainCar cannot reach the goal in 5 steps, so a limit of 5
        guarantees the episode ends by truncation, never termination."""
        key = jax.random.key(0)
        env = gxm.make("Gymnax/MountainCar-v0")
        adapter = env.get_wrapper(GymnaxAdapter)
        adapter.env_params = adapter.env_params.replace(max_steps_in_episode=5)
        env_state, timestep = env.init(key)
        for _ in range(5):
            key, key_step = jax.random.split(key)
            action = env.action_space.sample(key_step)
            env_state, timestep = env.step(key_step, env_state, action)

        assert bool(timestep.truncated)
        assert not bool(timestep.terminated)
        # next_obs is the auto-reset observation (fresh car: velocity == 0);
        # true_next_obs preserves the pre-reset observation (moving car).
        assert timestep.next_obs[1] == 0.0
        assert timestep.true_next_obs[1] != 0.0

    def test_termination_is_labeled(self):
        """Constantly pushing the CartPole left terminates (pole falls) long
        before the 500-step limit, so done must be labeled terminated."""
        key = jax.random.key(0)
        env = gxm.make("Gymnax/CartPole-v1")
        env_state, timestep = env.init(key)
        for _ in range(100):
            key, key_step = jax.random.split(key)
            env_state, timestep = env.step(key_step, env_state, jnp.int32(0))
            if bool(timestep.done):
                break

        assert bool(timestep.done), "CartPole did not terminate under constant push"
        assert bool(timestep.terminated)
        assert not bool(timestep.truncated)
