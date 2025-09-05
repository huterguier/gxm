from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class EnvironmentState:
    state: Any
    obs: jax.Array
    reward: float | jax.Array
    done: bool | jax.Array
    info: dict[str, Any]

    def __getitem__(self, item):
        return (self.state, self.obs, self.reward, self.done, self.info)[item]

    def __iter__(self):
        return iter((self.state, self.obs, self.reward, self.done, self.info))


class Environment:
    """Base class for environments in gxm."""

    def init(self, key: jax.Array) -> EnvironmentState:
        """Initialize the environment and return the initial state."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def step(
        self,
        key: jax.Array,
        env_state: EnvironmentState,
        action: jax.Array,
    ) -> EnvironmentState:
        """Perform a step in the environment given an action."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self, key: jax.Array, env_state: EnvironmentState) -> EnvironmentState:
        """Reset the environment to its initial state."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def num_actions(self) -> int:
        """Return the number of actions available in the environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")


@jax.tree_util.register_dataclass
@dataclass
class EpisodeStatistics:
    episodic_return: jax.Array
    discounted_episodic_return: jax.Array
    length: jax.Array
    _episodic_return: jax.Array
    _discounted_episodic_return: jax.Array
    _length: jax.Array


class RecordEpisodeStatistics:
    env: Environment
    gamma: float

    def __init__(self, env: Environment, gamma: float = 1.0):
        self.env = env
        self.gamma = gamma

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)

    def init(self, key: jax.Array) -> EnvironmentState:
        env_state = self.env.init(key)
        episode_stats = EpisodeStatistics(
            episodic_return=jnp.float32(0.0),
            discounted_episodic_return=jnp.float32(0.0),
            length=jnp.int32(0.0),
            _episodic_return=jnp.float32(0.0),
            _discounted_episodic_return=jnp.float32(0.0),
            _length=jnp.int32(0.0),
        )
        env_state = EnvironmentState(
            state=(env_state.state, episode_stats),
            obs=env_state.obs,
            reward=env_state.reward,
            done=env_state.done,
            info={
                "episodic_return": episode_stats.episodic_return,
                "discounted_episodic_return": episode_stats.discounted_episodic_return,
                "length": episode_stats.length,
            },
        )
        return env_state

    def reset(self, key: jax.Array, env_state: EnvironmentState) -> EnvironmentState:
        (state, episode_stats) = env_state.state
        env_state = EnvironmentState(
            state=state,
            obs=env_state.obs,
            reward=env_state.reward,
            done=env_state.done,
            info=env_state.info,
        )
        env_state = self.env.reset(key, env_state)
        episode_stats = EpisodeStatistics(
            episodic_return=jnp.float32(0.0),
            discounted_episodic_return=jnp.float32(0.0),
            length=jnp.int32(0.0),
            _episodic_return=jnp.float32(0.0),
            _discounted_episodic_return=jnp.float32(0.0),
            _length=jnp.int32(0.0),
        )
        env_state = EnvironmentState(
            state=(env_state, episode_stats),
            obs=env_state.obs,
            reward=env_state.reward,
            done=env_state.done,
            info={
                "episodic_return": episode_stats.episodic_return,
                "discounted_episodic_return": episode_stats.discounted_episodic_return,
                "length": episode_stats.length,
            },
        )
        return env_state

    def step(
        self,
        key: jax.Array,
        env_state: EnvironmentState,
        action: jax.Array,
    ) -> EnvironmentState:
        (state, episode_stats) = env_state.state
        env_state = EnvironmentState(
            state=state,
            obs=env_state.obs,
            reward=env_state.reward,
            done=env_state.done,
            info=env_state.info,
        )
        env_state = self.env.step(key, env_state, action)
        done = env_state.done
        reward = env_state.reward

        _episodic_return = episode_stats._episodic_return + reward
        _discounted_episodic_return = (
            episode_stats._discounted_episodic_return
            + reward * self.gamma**episode_stats._length
        )
        _length = episode_stats._length + 1

        episodic_return = (
            1 - done
        ) * episode_stats.episodic_return + done * _episodic_return
        discounted_episodic_return = (
            (1 - done) * episode_stats.discounted_episodic_return
            + done * _discounted_episodic_return
        )
        length = (1 - done) * episode_stats.length + done * _length

        _episodic_return = (1 - done) * _episodic_return
        _discounted_episodic_return = (1 - done) * _discounted_episodic_return
        _length = (1 - done) * _length

        episode_stats = EpisodeStatistics(
            episodic_return=episodic_return,
            discounted_episodic_return=discounted_episodic_return,
            length=length,
            _episodic_return=_episodic_return,
            _discounted_episodic_return=_discounted_episodic_return,
            _length=_length,
        )
        env_state = EnvironmentState(
            state=(env_state.state, episode_stats),
            obs=env_state.obs,
            reward=env_state.reward,
            done=env_state.done,
            info={
                "episodic_return": episode_stats.episodic_return,
                "discounted_episodic_return": episode_stats.discounted_episodic_return,
                "length": episode_stats.length,
            },
        )
        return env_state
