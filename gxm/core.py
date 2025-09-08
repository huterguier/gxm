from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_dataclass
@dataclass
class Timestep:
    """Class representing a single timestep in an environment."""

    obs: Array
    true_obs: Array
    reward: Array
    terminated: Array
    truncated: Array
    info: dict[str, Any]

    @property
    def done(self) -> Array:
        """Return whether the episode has ended (either terminated or truncated)."""
        return jnp.logical_or(self.terminated, self.truncated)

    def trajectory(self, first_obs: Any, action: Array) -> "Trajectory":
        """Convert a sequence of timesteps into a trajectory.
        Args:
            first_obs: The observation at the first timestep.
            action: The action taken at each timestep.
        Returns:
            A Trajectory object containing the sequence of timesteps.
        """
        assert (
            self.obs.shape[0] == action.shape[0]
        ), "The number of observations must match the number of actions."
        return Trajectory(
            obs=jnp.concatenate([first_obs[None], self.obs], axis=0),
            true_obs=jnp.concatenate([first_obs[None], self.true_obs], axis=0),
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            info=self.info,
            action=action,
        )


@jax.tree_util.register_dataclass
@dataclass
class Trajectory:
    """Class representing a trajectory of timesteps in an environment."""

    obs: Array
    true_obs: Array
    action: Array
    reward: Array
    terminated: Array
    truncated: Array
    info: dict[str, Any]

    def __len__(self):
        """Return the length of the trajectory."""
        assert (
            self.reward.ndim == 1
        ), "Trajectory length is only defined for batch size 1."
        return self.reward.shape[0]


@jax.tree_util.register_dataclass
@dataclass
class EnvironmentState(Timestep):
    """Class representing the state of an environment."""

    state: Any

    @property
    def timestep(self) -> Timestep:
        """Convert the EnvironmentState to a Timestep."""
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
