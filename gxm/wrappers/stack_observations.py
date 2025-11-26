from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class StackObservationsState(WrapperState):
    """State for the StackObservations wrapper."""

    obss: PyTree
    """The history of observations."""
    true_obss: PyTree
    """The history of true observations."""


class StackObservations(Wrapper[StackObservationsState]):
    """Wrapper that stacks the observation along a new axis."""

    num_stack: int
    padding: str

    def __init__(self, env: Environment, n_stack: int, padding: str = "reset"):
        self.env = env
        self.num_stack = n_stack
        self.padding = padding

    def init(self, key: Key) -> tuple[StackObservationsState, Timestep]:
        def stack(obss):
            return jax.tree.map(lambda *os: jnp.stack(os, axis=0), *obss)

        env_state, timestep = self.env.init(key)

        if self.padding == "reset":
            obss = stack(self.num_stack * [timestep.obs])
            true_obss = stack(self.num_stack * [timestep.true_obs])
            timestep.obs = obss
            timestep.true_obs = true_obss
        else:
            raise ValueError(f"Unknown padding method: {self.padding}")

        stack_observations_state = StackObservationsState(
            env_state=env_state, obss=obss, true_obss=true_obss
        )

        return stack_observations_state, timestep

    def reset(
        self, key: Key, env_state: StackObservationsState
    ) -> tuple[StackObservationsState, Timestep]:
        def stack(obss):
            return jax.tree.map(lambda *os: jnp.stack(os, axis=0), *obss)

        env_state, timestep = self.env.reset(key, env_state.env_state)

        if self.padding == "reset":
            obss = stack(self.num_stack * [timestep.obs])
            true_obss = stack(self.num_stack * [timestep.true_obs])
            timestep.obs = obss
            timestep.true_obs = true_obss
        else:
            raise ValueError(f"Unknown padding method: {self.padding}")

        stack_observations_state = StackObservationsState(
            env_state=env_state, obss=obss, true_obss=true_obss
        )

        return stack_observations_state, timestep

    def step(
        self,
        key: Key,
        env_state: StackObservationsState,
        action: PyTree,
    ) -> tuple[StackObservationsState, Timestep]:
        def concatenate(obss: Sequence[PyTree]) -> PyTree:
            return jax.tree.map(lambda *os: jnp.concatenate(os, axis=0), *obss)

        def expand_dims(obs: PyTree) -> PyTree:
            return jax.tree.map(lambda o: jnp.expand_dims(o, axis=0), obs)

        obss = env_state.obss
        true_obss = env_state.true_obss
        env_state, timestep = self.env.step(key, env_state.env_state, action)

        obss = jax.tree.map(lambda os: os[1:], obss)
        obss = concatenate([obss, expand_dims(timestep.obs)])
        true_obss = jax.tree.map(lambda tos: tos[1:], true_obss)
        true_obss = concatenate([true_obss, expand_dims(timestep.true_obs)])

        timestep.obs = obss
        timestep.true_obs = true_obss

        stack_observations_state = StackObservationsState(
            env_state=env_state,
            obss=obss,
            true_obss=true_obss,
        )

        return stack_observations_state, timestep
