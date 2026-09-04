import jax
import jax.numpy as jnp
import pytest
from test_wrapper import TestWrapper

import gxm
from gxm.core import Environment
from gxm.wrappers import SquashActions, Wrapper


class TestSquashActions(TestWrapper):
    @pytest.fixture(params=["Gymnax/Pendulum-v1", "Gymnax/MountainCarContinuous-v0"])
    def env(self, request) -> Environment:
        return gxm.make(request.param)

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return SquashActions(env)

    def test_step_keeps_unsquashed_action(self, wrapper):
        key = jax.random.key(0)
        state, _ = wrapper.init(key)
        action = jnp.full(wrapper.action_space.shape, 10.0)
        state, timestep = wrapper.step(key, state, action)
        assert jnp.allclose(timestep.action, action)

    def test_squash_maps_onto_bounds(self, wrapper):
        low, high = wrapper.action_space.low, wrapper.action_space.high
        assert jnp.allclose(wrapper.squash(jnp.full(low.shape, 0.0)), (low + high) / 2)
        assert jnp.allclose(wrapper.squash(jnp.full(low.shape, 100.0)), high)
        assert jnp.allclose(wrapper.squash(jnp.full(low.shape, -100.0)), low)

    def test_rejects_discrete_action_space(self, dummy_dynamics):
        with pytest.raises(TypeError):
            SquashActions(dummy_dynamics)
