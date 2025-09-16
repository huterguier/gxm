import jax.numpy as jnp
from jax import Array

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.wrappers.wrapper import Wrapper


class StackObservations(Wrapper):
    """Wrapper that stacks the observation along a new axis."""

    num_stack: int
    padding: str

    def __init__(self, env: Environment, num_stack: int, padding: str = "reset"):
        self.env = env
        self.num_stack = num_stack
        self.padding = padding

    def init(self, key: Array) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.init(key)
        if self.padding == "reset":
            timestep.obs = jnp.stack(self.num_stack * [timestep.obs], axis=0)
            timestep.true_obs = jnp.stack(self.num_stack * [timestep.true_obs], axis=0)
        else:
            raise ValueError(f"Unknown padding method: {self.padding}")
        env_state = (env_state, (timestep.obs, timestep.true_obs))

        return env_state, timestep

    def reset(
        self, key: Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        if self.padding == "reset":
            env_state.obs = jnp.stack(self.num_stack * [env_state.obs], axis=0)
            env_state.true_obs = jnp.stack(
                self.num_stack * [env_state.true_obs], axis=0
            )
        else:
            raise ValueError(f"Unknown padding method: {self.padding}")
        return env_state, timestep

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.step(key, env_state, action)
        timestep.obs = jnp.concatenate(
            [timestep.obs[1:], jnp.expand_dims(timestep.obs[0], axis=0)], axis=0
        )
        timestep.true_obs = jnp.concatenate(
            [timestep.true_obs[1:], jnp.expand_dims(timestep.true_obs[0], axis=0)],
            axis=0,
        )
        return env_state, timestep
