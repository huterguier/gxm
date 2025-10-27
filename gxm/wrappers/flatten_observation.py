import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


class FlattenObservation(Wrapper):
    """Wrapper that adds a rollout method to the environment."""

    def __init__(self, env: Environment, unwrap: bool = True):
        super().__init__(env, unwrap=unwrap)

    @classmethod
    def flatten(cls, obs: PyTree) -> Array:
        obs_leaves = jax.tree.leaves(obs)
        obs_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in obs_leaves])
        return obs_flat

    def init(self, key: Key) -> tuple[WrapperState, Timestep]:
        env_state, timestep = self.env.init(key)
        timestep.obs = self.flatten(timestep.obs)
        return env_state, timestep

    def reset(self, key: Key, env_state: WrapperState) -> tuple[WrapperState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        timestep.obs = self.flatten(timestep.obs)
        return env_state, timestep

    def step(
        self,
        key: Key,
        env_state: WrapperState,
        action: PyTree,
    ) -> tuple[WrapperState, Timestep]:
        env_state, timestep = self.env.step(key, env_state, action)
        timestep.obs = self.flatten(timestep.obs)
        return env_state, timestep
