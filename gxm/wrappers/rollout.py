from typing import Callable

import jax
from jax import Array

from gxm.core import Environment, EnvironmentState, Trajectory
from gxm.wrappers.wrapper import Wrapper


class Rollout(Wrapper):
    """Wrapper that adds a rollout method to the environment."""

    def __init__(self, env: Environment):
        self.env = env

    def init(self, key: Array) -> EnvironmentState:
        return self.env.init(key)

    def reset(self, key: Array, env_state: EnvironmentState) -> EnvironmentState:
        return self.env.reset(key, env_state)

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> EnvironmentState:
        return self.env.step(key, env_state, action)

    def rollout(
        self,
        key: Array,
        env_state: EnvironmentState,
        pi: Callable[[Array, EnvironmentState], Array],
        num_steps: int,
    ) -> tuple[EnvironmentState, Trajectory]:
        def _step(carry, key):
            key, env_state = carry
            key_pi, key_step = jax.random.split(key)
            action = pi(key_pi, env_state)
            env_state = self.step(key_step, env_state, action)
            trajectory = env_state.trajectory(action)
            return env_state, trajectory

        keys = jax.random.split(key, num_steps)
        env_state, trajectory = jax.lax.scan(
            _step,
            env_state,
            keys,
        )
        return env_state, trajectory
