from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_dataclass
@dataclass
class Timestep:
    """Class representing a single timestep in an environment."""

    reward: Array
    terminated: Array
    truncated: Array
    obs: Array
    true_obs: Array
    info: dict[str, Any]

    @property
    def done(self) -> Array:
        """Return whether the episode has ended (either terminated or truncated)."""
        return jnp.logical_or(self.terminated, self.truncated)

    def transition(
        self,
        prev_obs: Any,
        action: Array,
        prev_info: dict[str, Any] = {},
    ) -> "Transition":
        """Convert the current timestep :math:`(R_t, S_{t+1})` into a transition
        :math:`(S_t, A_t, R_t, S_{t+1})` given the previous observation :math:`S_t`
        and the action :math:`A_t`.
        Args:
            prev_obs: The observation at the previous timestep.
            action: The action taken at the current timestep.
            prev_info: The info at the previous timestep.
        Returns:
            A Transition object containing the current and next timesteps.
        """
        return Transition(
            prev_obs=prev_obs,
            prev_info=prev_info,
            action=action,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            obs=self.obs,
            info=self.info,
        )

    def trajectory(
        self, first_obs: Any, action: Array, first_info: dict[str, Any] = {}
    ) -> "Trajectory":
        r"""
        Convert a sequence of timesteps :math:`(R_0, S_1, ..., S_n)` with
        the first observation :math:`S_0` and the actions :math:`(A_0, A_1, ..., A_{n-1})`
        into a trajectory :math:`(S_0, A_0, R_0, S_1, ..., S_n)`.

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
class Transition:
    """Class representing a single transition in an environment."""

    prev_obs: Array
    prev_info: dict[str, Any]
    action: Array
    reward: Array
    terminated: Array
    truncated: Array
    obs: Array
    info: dict[str, Any]

    @property
    def done(self) -> Array:
        """Return whether the episode has ended (either terminated or truncated)."""
        return jnp.logical_or(self.terminated, self.truncated)


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

    @property
    def done(self) -> Array:
        """Return whether the episode has ended (either terminated or truncated)."""
        return jnp.logical_or(self.terminated, self.truncated)

    def __len__(self):
        """Return the length of the trajectory."""
        assert (
            self.reward.ndim == 1
        ), "Trajectory length is only defined for batch size 1."
        return self.reward.shape[0]


EnvironmentState = Any


class Environment:
    """Base class for environments in gxm."""

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        """Initialize the environment and return the initial state."""
        del key
        raise NotImplementedError("This method should be implemented by subclasses.")

    def step(
        self,
        key: jax.Array,
        env_state: EnvironmentState,
        action: jax.Array,
    ) -> tuple[EnvironmentState, Timestep]:
        """Perform a step in the environment given an action."""
        del key, env_state, action
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(
        self, key: jax.Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        """Reset the environment to its initial state."""
        del key, env_state
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def num_actions(self) -> int:
        """Return the number of actions available in the environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")
