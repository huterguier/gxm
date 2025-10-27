from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp

from gxm.spaces import Space
from gxm.typing import Array, Key, PyTree


@jax.tree_util.register_dataclass
@dataclass
class Timestep:
    """
    Class representing a single timestep :math:`(R_i, S_{i+1})` in an environment.
    Where :math:`R_i` is the reward received after taking an action at timestep
    :math:`i` and :math:`S_{i+1}` is the observation at the next timestep.
    In case of truncation, ``true_obs`` represents the observation :math:`\hat{S}_{i+1}` that would have been
    observed if the episode had not been truncated.
    """

    reward: Array
    """The reward :math:`R_i` received at this timestep :math:`i`."""
    terminated: Array
    """Whether the episode has terminated at this timestep."""
    truncated: Array
    """Whether the episode has been truncated at this timestep."""
    obs: PyTree
    """The observation :math:`S_{i+1}` at this timestep :math:`i`."""
    true_obs: PyTree
    """The true observation :math:`\hat{S}_{i+1}` at this timestep. This may differ from ``obs`` in environments that allow truncation, if and only if truncation is True. """
    info: dict[str, PyTree]
    """Additional information about the timestep."""

    @property
    def done(self) -> Array:
        """Whether the episode has terminated or been truncated."""
        return jnp.logical_or(self.terminated, self.truncated)

    def transition(
        self,
        prev_obs: PyTree,
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
        self, first_obs: PyTree, action: Array, first_info: dict[str, PyTree] = {}
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
    """Class representing a single transition :math:`(S_i, A_i, R_i, S_{i+1})` in an environment."""

    prev_obs: PyTree
    prev_info: dict[str, PyTree]
    action: PyTree
    reward: Array
    terminated: Array
    truncated: Array
    obs: PyTree
    info: dict[str, PyTree]

    @property
    def done(self) -> Array:
        """Return whether the episode has ended (either terminated or truncated)."""
        return jnp.logical_or(self.terminated, self.truncated)


@jax.tree_util.register_dataclass
@dataclass
class Trajectory:
    """Class representing a trajectory :math:`(S_0, A_0, R_0, S_1, ..., S_n)` in an environment."""

    obs: PyTree
    """The observations :math:`(S_0, S_1, ..., S_n)` in the trajectory."""
    true_obs: PyTree
    """The true observations :math:`(\hat{S}_0, \hat{S}_1, ..., \hat{S}_n)` in the trajectory. These may differ from ``obs`` in environments that allow truncation."""
    action: PyTree
    """The actions :math:`(A_0, A_1, ..., A_{n-1})` taken in the trajectory."""
    reward: Array
    """The rewards :math:`(R_0, R_1, ..., R_{n-1})` received in the trajectory."""
    terminated: Array
    """Whether the episode terminated at each timestep in the trajectory."""
    truncated: Array
    """Whether the episode was truncated at each timestep in the trajectory."""
    info: dict[str, PyTree]
    """Additional information about the trajectory."""

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


class EnvironmentState:
    """
    A placeholder class for environment state.
    This can be replaced with a more specific implementation as needed.
    """

    pass


TEnvironmentState = TypeVar("TEnvironmentState", bound=EnvironmentState)


class Environment(Generic[TEnvironmentState], ABC):
    """
    Base class for environments in ``gxm``.
    Environments should inherit from this class and implement the
    ``init``, ``step``, ``reset``, and ``num_actions`` methods.
    """

    id: str
    """The unique identifier of the environment."""
    action_space: Space
    """The action space of the environment."""
    observation_space: Space
    """The observation space of the environment."""

    @abstractmethod
    def init(self, key: Key) -> tuple[TEnvironmentState, Timestep]:
        """
        Initialize the environment and return the initial state.

        Args:
            key: A JAX random key for any stochastic initialization.
        Returns:
            A tuple containing the initial environment state and the initial timestep.
        """

    @abstractmethod
    def reset(
        self, key: Key, env_state: TEnvironmentState
    ) -> tuple[TEnvironmentState, Timestep]:
        """
        Reset the environment to its initial state.

        Args:
            key: A JAX random key for any stochasticity in the environment.
            env_state: The current state of the environment.
        Returns:
            A tuple containing the reset environment state and the initial timestep.
        """
        pass

    @abstractmethod
    def step(
        self,
        key: Key,
        env_state: TEnvironmentState,
        action: PyTree,
    ) -> tuple[TEnvironmentState, Timestep]:
        """
        Perform a step in the environment given an action.

        Args:
            key: A JAX random key for any stochasticity in the environment.
            env_state: The current state of the environment.
            action: The action to take in the environment.
        Returns:
            A tuple containing the new environment state and the resulting timestep.
        """
        pass

    def has_wrapper(self, wrapper_type: type["Environment"]) -> bool:
        """
        Check if the environment or any of its wrappers is of a specific type.

        Args:
            wrapper_type: The type to check for.
        Returns:
            True if the environment or any of its wrappers is of the specified type, False otherwise.
        """
        return isinstance(self, wrapper_type)

    def get_wrapper(self, wrapper_type: type["Environment"]) -> "Environment":
        """
        Retrieve the first wrapper of a specific type from the environment.

        Args:
            wrapper_type: The type of the wrapper to retrieve.
        Returns:
            The first wrapper of the specified type.
        Raises:
            ValueError: If no wrapper of the specified type is found.
        """
        if isinstance(self, wrapper_type):
            return self
        raise ValueError(f"No wrapper of type {wrapper_type} found in the environment.")

    @property
    def unwrapped(self) -> "Environment":
        """
        Retrieve the base environment by unwrapping all wrappers.

        Returns:
            The base environment without any wrappers.
        """
        return self
