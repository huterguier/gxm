import jax
import pytest

import gxm
from gxm.core import Environment
from gxm.wrappers import Wrapper


class TestWrapper:

    @pytest.fixture(
        params=[
            "Gymnax/CartPole-v1",
            "Gymnax/Breakout-MinAtar",
        ]
    )
    def env(self, request) -> Environment:
        return gxm.make(request.param)

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        pytest.skip("Base Wrapper class cannot be instantiated directly.")

    def test_init(self, wrapper: Wrapper):
        key = jax.random.key(0)
        wrapper_state, timestep = wrapper.init(key)

    def test_reset(self, wrapper: Wrapper):
        key = jax.random.key(0)
        wrapper_state, timestep = wrapper.init(key)
        wrapper_state, timestep = wrapper.reset(key, wrapper_state)

    def test_step(self, wrapper: Wrapper):
        key = jax.random.key(0)
        wrapper_state, timestep = wrapper.init(key)
        action = wrapper.action_space.sample(key)
        wrapper_state, timestep = wrapper.step(key, wrapper_state, action)

    def test_vmap_init(self, wrapper: Wrapper):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        wrapper_states, timesteps = jax.vmap(wrapper.init)(keys)

    def test_vmap_reset(self, wrapper: Wrapper):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        wrapper_states, timesteps = jax.vmap(wrapper.init)(keys)
        wrapper_states, timesteps = jax.vmap(wrapper.reset)(keys, wrapper_states)

    def test_vmap_step(self, wrapper: Wrapper):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        wrapper_states, timesteps = jax.vmap(wrapper.init)(keys)
        actions = wrapper.action_space.sample(key, (10,))
        wrapper_states, timesteps = jax.vmap(wrapper.step)(
            keys, wrapper_states, actions
        )

    def test_scan_step(self, wrapper: Wrapper):
        key = jax.random.key(0)
        wrapper_state, _ = wrapper.init(key)

        def step_fn(carry, _):
            wrapper_state = carry
            action = wrapper.action_space.sample(key)
            wrapper_state, timestep = wrapper.step(key, wrapper_state, action)
            return wrapper_state, timestep

        jax.lax.scan(step_fn, wrapper_state, length=10)
