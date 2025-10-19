from typing import Any, Sequence

import jax
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

        def stack(obss):
            return jax.tree.map(lambda *os: jnp.stack(os, axis=0), *obss)

        if self.padding == "reset":
            timestep.obs = stack([self.num_stack * [timestep.obs]])
            timestep.true_obs = stack([self.num_stack * [timestep.true_obs]])
        else:
            raise ValueError(f"Unknown padding method: {self.padding}")

        return (env_state, (timestep.obs, timestep.true_obs)), timestep

    def reset(
        self, key: Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)

        def stack(obss):
            return jax.tree.map(lambda *os: jnp.stack(os, axis=0), *obss)

        if self.padding == "reset":
            timestep.obs = stack([self.num_stack * [timestep.obs]])
            timestep.true_obs = stack([self.num_stack * [timestep.true_obs]])
        else:
            raise ValueError(f"Unknown padding method: {self.padding}")
        return (env_state, (timestep.obs, timestep.true_obs)), timestep

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, (obss, true_obss) = env_state
        env_state, timestep = self.env.step(key, env_state, action)
        obss = jax.tree.map(lambda os: os[1:], obss)
        true_obss = jax.tree.map(lambda tos: tos[1:], true_obss)

        def concatenate(obss: Sequence[Any]) -> Any:
            return jax.tree.map(lambda *os: jnp.concatenate(os, axis=0), *obss)

        def expand_dims(obs: Any) -> Any:
            return jax.tree.map(lambda o: jnp.expand_dims(o, axis=0), obs)

        timestep.obs = concatenate([obss, expand_dims(timestep.obs)])
        timestep.true_obs = concatenate([true_obss, expand_dims(timestep.true_obs)])

        return (env_state, (timestep.obs, timestep.true_obs)), timestep
