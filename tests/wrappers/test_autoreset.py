from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from test_wrapper import TestWrapper

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete
from gxm.typing import Key, PyTree
from gxm.wrappers import AutoReset, Wrapper


@jax.tree_util.register_dataclass
@dataclass
class CountingEnvState(EnvironmentState):
    count: jnp.ndarray


class CountingEnv(Environment[CountingEnvState]):
    """
    A deterministic environment whose observation is its own step count, so
    AutoReset's blending behavior can be asserted on exact values rather than
    just shapes.
    """

    id = "CountingEnv"
    action_space = Discrete(1)
    observation_space = Box(low=0.0, high=jnp.inf, shape=())

    def __init__(self, limit: int, truncate: bool = False):
        self.limit = limit
        self.truncate = truncate

    def init(self, key: Key) -> tuple[CountingEnvState, Timestep]:
        return self.reset(key, CountingEnvState(count=jnp.float32(0.0)))

    def reset(
        self, key: Key, state: CountingEnvState
    ) -> tuple[CountingEnvState, Timestep]:
        state = CountingEnvState(count=jnp.float32(0.0))
        timestep = Timestep(
            next_obs=state.count,
            true_next_obs=state.count,
            action=self.action_space.sample(key),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(False),
            truncated=jnp.bool(False),
            info={},
        )
        return state, timestep

    def step(
        self, key: Key, state: CountingEnvState, action: PyTree
    ) -> tuple[CountingEnvState, Timestep]:
        count = state.count + 1.0
        done = count >= self.limit
        state = CountingEnvState(count=count)
        timestep = Timestep(
            next_obs=count,
            true_next_obs=count,
            action=action,
            reward=jnp.float32(1.0),
            terminated=jnp.bool(False) if self.truncate else done,
            truncated=done if self.truncate else jnp.bool(False),
            info={},
        )
        return state, timestep


class TestAutoReset(TestWrapper):
    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        return AutoReset(env)

    def test_resets_on_termination(self):
        key = jax.random.key(0)
        env = CountingEnv(limit=3)
        wrapper = AutoReset(env)
        state, timestep = wrapper.init(key)

        counts = []
        terminated = []
        for _ in range(6):
            state, timestep = wrapper.step(key, state, jnp.int32(0))
            counts.append(int(timestep.next_obs))
            terminated.append(bool(timestep.terminated))

        assert counts == [1, 2, 0, 1, 2, 0]
        assert terminated == [False, False, True, False, False, True]
        # On termination the agent acts on the reset obs, but true_next_obs
        # preserves the terminal observation the environment actually produced.
        assert int(timestep.next_obs) == 0
        assert int(timestep.true_next_obs) == 3

    def test_true_next_obs_on_truncation(self):
        key = jax.random.key(0)
        env = CountingEnv(limit=3, truncate=True)
        wrapper = AutoReset(env)
        state, timestep = wrapper.init(key)

        for _ in range(3):
            state, timestep = wrapper.step(key, state, jnp.int32(0))

        assert bool(timestep.truncated)
        assert not bool(timestep.terminated)
        # The observation seen by the agent is the reset (0)...
        assert int(timestep.next_obs) == 0
        # ...but true_next_obs preserves what would have been seen absent reset.
        assert int(timestep.true_next_obs) == 3
