import jax
import pytest

from gxm.core import Environment


class TestEnvironment:
    @pytest.fixture(params=[])
    def env(request) -> Environment:
        raise NotImplementedError("Add environments to the fixture parameters.")

    def test_init(self, env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        assert env_state is not None
        assert timestep is not None

    def test_reset(self, env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        env_state, timestep = env.reset(key, env_state)
        assert env_state is not None
        assert timestep is not None

    def test_step(self, env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        action = env.action_space.sample(key)
        env_state, timestep = env.step(key, env_state, action)
        assert env_state is not None
        assert timestep is not None

    def test_vmap_init(self, env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        assert env_states is not None
        assert timesteps is not None

    def test_vmap_reset(self, env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        env_states, timesteps = jax.vmap(env.reset)(keys, env_states)
        assert env_states is not None
        assert timesteps is not None

    def test_vmap_step(self, env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        actions = env.action_space.sample(key, (10,))
        env_states, timesteps = jax.vmap(env.step)(keys, env_states, actions)
        assert env_states is not None
        assert timesteps is not None
