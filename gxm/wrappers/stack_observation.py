import jax.numpy as jnp
from jax import Array

from gxm.core import Environment, EnvironmentState
from gxm.wrappers.wrapper import Wrapper


class StackObservation(Wrapper):
    """Wrapper that stacks the observation along a new axis."""

    num_stack: int
    padding: str

    def __init__(self, env: Environment, num_stack: int, padding: str = "reset"):
        self.env = env
        self.num_stack = num_stack
        self.padding = padding

    def init(self, key: Array) -> EnvironmentState:
        env_state = self.env.init(key)
        if self.padding == "reset":
            env_state.obs = jnp.stack(self.num_stack * [env_state.obs], axis=0)
            env_state.true_obs = jnp.stack(
                self.num_stack * [env_state.true_obs], axis=0
            )
        else:
            raise ValueError(f"Unknown padding method: {self.padding}")

        return env_state

    def reset(self, key: Array, env_state: EnvironmentState) -> EnvironmentState:
        env_state = self.env.reset(key, env_state)
        if self.padding == "reset":
            env_state.obs = jnp.stack(self.num_stack * [env_state.obs], axis=0)
            env_state.true_obs = jnp.stack(
                self.num_stack * [env_state.true_obs], axis=0
            )
        else:
            raise ValueError(f"Unknown padding method: {self.padding}")
        return env_state

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> EnvironmentState:
        env_state = self.env.step(key, env_state, action)
        env_state.obs = jnp.concatenate(
            [env_state.obs[1:], jnp.expand_dims(env_state.obs[0], axis=0)], axis=0
        )
        env_state.true_obs = jnp.concatenate(
            [env_state.true_obs[1:], jnp.expand_dims(env_state.true_obs[0], axis=0)],
            axis=0,
        )
        return env_state
