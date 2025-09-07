from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_dataclass
@dataclass
class Timestep:
    obs: Array
    true_obs: Array
    reward: Array
    terminated: Array
    truncated: Array
    info: dict[str, Any]

    @property
    def done(self) -> Array:
        return jnp.logical_or(self.terminated, self.truncated)

    def trajectory(self, action: Array) -> "Trajectory":
        return Trajectory(
            obs=self.obs,
            true_obs=self.obs,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            info=self.info,
            action=action,
        )


@jax.tree_util.register_dataclass
@dataclass
class Trajectory(Timestep):
    action: Array

    def __len__(self):
        assert (
            self.done.ndim == 1
        ), "Trajectory length is only defined for batch size 1."
        return self.done.shape[0]


@jax.tree_util.register_dataclass
@dataclass
class EnvironmentState(Timestep):
    state: Any

    @property
    def timestep(self) -> Timestep:
        return Timestep(
            obs=self.obs,
            true_obs=self.obs,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            info=self.info,
        )


class Environment:
    """Base class for environments in gxm."""

    def init(self, key: jax.Array) -> EnvironmentState:
        """Initialize the environment and return the initial state."""
        del key
        raise NotImplementedError("This method should be implemented by subclasses.")

    def step(
        self,
        key: jax.Array,
        env_state: EnvironmentState,
        action: jax.Array,
    ) -> EnvironmentState:
        """Perform a step in the environment given an action."""
        del key, env_state, action
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self, key: jax.Array, env_state: EnvironmentState) -> EnvironmentState:
        """Reset the environment to its initial state."""
        del key, env_state
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def num_actions(self) -> int:
        """Return the number of actions available in the environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")
