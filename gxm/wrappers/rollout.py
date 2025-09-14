from typing import Callable

import jax
from jax import Array

from gxm.core import Environment, EnvironmentState, Timestep, Trajectory
from gxm.wrappers.wrapper import Wrapper


class Rollout(Wrapper):
    """Wrapper that adds a rollout method to the environment."""

    def __init__(self, env: Environment):
        self.env = env

    def init(self, key: Array) -> tuple[EnvironmentState, Timestep]:
        return self.env.init(key)

    def reset(
        self, key: Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        return self.env.reset(key, env_state)

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> tuple[EnvironmentState, Timestep]:
        return self.env.step(key, env_state, action)

    def rollout(
        self,
        key: Array,
        env_state: EnvironmentState,
        pi: Callable[[Array, Array], Array],
        num_steps: int,
    ) -> tuple[EnvironmentState, Trajectory]:
        def _step(carry, key):
            env_state, prev_obs = carry
            key_pi, key_step = jax.random.split(key)
            action = pi(
                key_pi,
                prev_obs,
            )
            env_state, timestep = self.step(key_step, env_state, action)
            return (env_state, timestep.obs), (env_state.timestep, action)

        first_obs = env_state.obs
        keys = jax.random.split(key, num_steps)
        env_state, (timesteps, actions) = jax.lax.scan(
            _step,
            env_state,
            keys,
        )
        trajectory = timesteps.trajectory(first_obs, actions)
        return env_state, trajectory
