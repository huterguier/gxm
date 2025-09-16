import jax.numpy as jnp
from jax import Array

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.wrappers.wrapper import Wrapper


class IgnoreTruncation(Wrapper):
    """
    A wrapper that treats truncation as termination and removes the corresponding obsercation from the timestep.

    >>> import gxm
    >>> from gxm.wrappers import IgnoreTruncation
    >>> env = make("Gymnax/CartPole-v1")
    >>> env = IgnoreTruncation(env)

    """

    env: Environment

    def __init__(self, env: Environment, actions: Array):
        """
        Args:
            env: The environment to wrap.
        """
        self.env = env

    def init(self, key: Array) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.init(key)
        timestep.truncated = None  # type: ignore
        timestep.true_obs = None  # type: ignore

        return env_state, timestep

    def reset(
        self, key: Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state)
        timestep.truncated = None  # type: ignore
        timestep.true_obs = None  # type: ignore
        return env_state, timestep

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> tuple[EnvironmentState, Timestep]:

        env_state, timestep = self.env.step(key, env_state, action)
        timestep.terminated = jnp.logical_or(timestep.terminated, timestep.truncated)
        timestep.truncated = None  # type: ignore
        timestep.true_obs = None  # type: ignore
        return env_state, timestep
