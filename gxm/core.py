import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp

from gxm.spaces import Space
from gxm.typing import Array, Key, PyTree


@jax.tree_util.register_dataclass
@dataclass
class Step:
    """
    Output of a single model step: a pure dynamics transition.

    Contains the next observation, the action taken, and any auxiliary info,
    but no episodic metadata (reward, termination, truncation).
    """

    next_obs: PyTree
    """The observation at the next state."""
    action: PyTree
    """The action taken at this step."""
    info: dict[str, PyTree]
    """Additional information about the step."""


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "next_obs",
        "true_next_obs",
        "action",
        "reward",
        "terminated",
        "truncated",
        "info",
    ],
    meta_fields=[],
)
@dataclass
class Timestep(Step):
    """
    Extends :class:`Step` with episodic metadata.

    The "time" refers to *episodic* time: reward, termination, and truncation
    signals only exist in the context of an episode, and are meaningless for a
    pure world model. :class:`Timestep` represents one unit of time
    :math:`(R_i, S_{i+1})` within a bounded episode.

    In case of truncation, ``true_next_obs`` holds the observation
    :math:`\\hat{S}_{i+1}` that would have been seen had the episode not been cut.
    """

    reward: Array
    """The reward :math:`R_i` received at this timestep."""
    terminated: Array
    """Whether the episode has terminated at this timestep."""
    truncated: Array
    """Whether the episode has been truncated at this timestep."""
    true_next_obs: PyTree
    """The true next observation before any auto-reset. Differs from ``next_obs`` only when ``truncated`` is True."""

    @property
    def done(self) -> Array:
        """Whether the episode has terminated or been truncated."""
        return jnp.logical_or(self.terminated, self.truncated)

    def transition(
        self,
        obs: PyTree,
    ) -> "Transition":
        """Convert the current timestep :math:`(R_t, S_{t+1})` into a transition
        :math:`(S_t, A_t, R_t, S_{t+1})` given the previous observation :math:`S_t`.
        Args:
            obs: The observation at the previous timestep.
        Returns:
            A Transition object containing the current and next timesteps.
        """
        return Transition(
            obs=obs,
            action=self.action,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            next_obs=self.next_obs,
            info=self.info,
        )

    def trajectory(self, first_obs: PyTree) -> "Trajectory":
        r"""
        Convert a sequence of timesteps :math:`(R_0, S_1, ..., S_n)` with
        the first observation :math:`S_0` into a trajectory :math:`(S_0, A_0, R_0, S_1, ..., S_n)`.

        Args:
            first_obs: The observation at the first timestep.
        Returns:
            A Trajectory object containing the sequence of timesteps.
        """
        return Trajectory(
            obs=jax.tree.map(
                lambda f, n: jnp.concatenate([f[None], n], axis=0),
                first_obs,
                self.next_obs,
            ),
            true_obs=jax.tree.map(
                lambda f, n: jnp.concatenate([f[None], n], axis=0),
                first_obs,
                self.true_next_obs,
            ),
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            info=self.info,
            action=self.action,
        )


@jax.tree_util.register_dataclass
@dataclass
class Transition:
    """Class representing a single transition :math:`(S_i, A_i, R_i, S_{i+1})` in an environment."""

    obs: PyTree
    action: PyTree
    reward: Array
    terminated: Array
    truncated: Array
    next_obs: PyTree
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
    """The true observations :math:`(\\hat{S}_0, \\hat{S}_1, ..., \\hat{S}_n)` in the trajectory. These may differ from ``obs`` in environments that allow truncation."""
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


class DynamicsState:
    """
    A placeholder class for dynamics/environment state.
    This can be replaced with a more specific implementation as needed.
    """

    pass


EnvironmentState = DynamicsState

TDynamicsState = TypeVar("TDynamicsState", bound=DynamicsState)
TEnvironmentState = TypeVar("TEnvironmentState", bound=DynamicsState)
TStep = TypeVar("TStep", bound=Step, covariant=True)


class Dynamics(Protocol[TDynamicsState, TStep]):
    """
    Base class for world models in ``gxm``.

    Dynamics define state transitions: given an action, they transition to a new
    state and produce a step output. They have no notion of episodes, rewards, or
    termination — those are added by :class:`Environment`.

    All :class:`Environment` instances are also ``Dynamics`` instances, so any
    function typed ``dynamics: Dynamics`` can accept an environment directly.
    """

    id: str
    """The unique identifier of the dynamics."""
    action_space: Space
    """The action space of the dynamics."""
    observation_space: Space
    """The observation space of the dynamics."""

    @abstractmethod
    def init(self, key: Key) -> tuple[TDynamicsState, TStep]:
        """
        Initialize the dynamics and return the initial state.

        Args:
            key: A JAX random key for any stochastic initialization.
        Returns:
            A tuple of the initial state and the initial step output.
        """

    @abstractmethod
    def reset(self, key: Key, state: TDynamicsState) -> tuple[TDynamicsState, TStep]:
        """
        Reset the dynamics to an initial state.

        Args:
            key: A JAX random key for any stochasticity.
            state: The current state.
        Returns:
            A tuple of the reset state and the initial step output.
        """

    @abstractmethod
    def step(
        self, key: Key, state: TDynamicsState, action: PyTree
    ) -> tuple[TDynamicsState, TStep]:
        """
        Advance the dynamics by one step given an action.

        Args:
            key: A JAX random key for any stochasticity.
            state: The current state.
            action: The action to apply.
        Returns:
            A tuple of the new state and the resulting step output.
        """

    def has_wrapper(self, wrapper_type: type["Dynamics"]) -> bool:
        """
        Check if the dynamics or any of its wrappers is of a specific type.

        Args:
            wrapper_type: The type to check for.
        Returns:
            True if the dynamics or any of its wrappers is of the specified type, False otherwise.
        """
        return isinstance(self, wrapper_type)

    def get_wrapper(self, wrapper_type: type["Dynamics"]) -> "Dynamics":
        """
        Retrieve the first wrapper of a specific type from the dynamics.

        Args:
            wrapper_type: The type of the wrapper to retrieve.
        Returns:
            The first wrapper of the specified type.
        Raises:
            ValueError: If no wrapper of the specified type is found.
        """
        if isinstance(self, wrapper_type):
            return self
        raise ValueError(f"No wrapper of type {wrapper_type} found in the dynamics.")

    @property
    def unwrapped(self) -> "Dynamics":
        """
        Retrieve the base dynamics by unwrapping all wrappers.

        Returns:
            The base dynamics without any wrappers.
        """
        return self


class Environment(Dynamics[TEnvironmentState, Timestep], Protocol[TEnvironmentState]):
    """
    Base class for RL environments in ``gxm``.

    Extends :class:`Dynamics` with episode structure: each step returns a
    :class:`Timestep` that includes reward, termination, and truncation signals.
    Environments should inherit from this class and implement
    ``init``, ``step``, and ``reset``.
    """

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
        self, key: Key, state: TEnvironmentState
    ) -> tuple[TEnvironmentState, Timestep]:
        """
        Reset the environment to its initial state.

        Args:
            key: A JAX random key for any stochasticity in the environment.
            state: The current state of the environment.
        Returns:
            A tuple containing the reset environment state and the initial timestep.
        """

    @abstractmethod
    def step(
        self,
        key: Key,
        state: TEnvironmentState,
        action: PyTree,
    ) -> tuple[TEnvironmentState, Timestep]:
        """
        Perform a step in the environment given an action.

        Args:
            key: A JAX random key for any stochasticity in the environment.
            state: The current state of the environment.
            action: The action to take in the environment.
        Returns:
            A tuple containing the new environment state and the resulting timestep.
        """


class AutoResetEnvironment(Generic[TEnvironmentState], Environment[TEnvironmentState]):
    """
    Base class for native gxm environments.
    Subclasses implement ``_reset`` and ``_step``; auto-reset on episode end is
    handled automatically in ``step``.
    """

    @abstractmethod
    def _reset(self, key: Key) -> tuple[TEnvironmentState, Timestep]:
        pass

    @abstractmethod
    def _step(
        self, key: Key, state: TEnvironmentState, action: PyTree
    ) -> tuple[TEnvironmentState, Timestep]:
        pass

    def init(self, key: Key) -> tuple[TEnvironmentState, Timestep]:
        return self._reset(key)

    def reset(
        self, key: Key, state: TEnvironmentState
    ) -> tuple[TEnvironmentState, Timestep]:
        return self._reset(key)

    def step(
        self, key: Key, state: TEnvironmentState, action: PyTree
    ) -> tuple[TEnvironmentState, Timestep]:
        key_step, key_reset = jax.random.split(key)
        state_step, timestep_step = self._step(key_step, state, action)
        state_reset, timestep_reset = self._reset(key_reset)
        state = jax.tree.map(
            lambda x_step, x_reset: jnp.where(timestep_step.done, x_reset, x_step),
            state_step,
            state_reset,
        )
        obs = jax.tree.map(
            lambda x_step, x_reset: jnp.where(timestep_step.done, x_reset, x_step),
            timestep_step.next_obs,
            timestep_reset.next_obs,
        )
        true_obs = jax.tree.map(
            lambda x_step, x_obs: jnp.where(timestep_step.truncated, x_step, x_obs),
            timestep_step.next_obs,
            obs,
        )
        return state, Timestep(
            next_obs=obs,
            true_next_obs=true_obs,
            action=action,
            reward=timestep_step.reward,
            terminated=timestep_step.terminated,
            truncated=timestep_step.truncated,
            info=timestep_step.info,
        )
