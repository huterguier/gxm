from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState
from gxm.wrappers.wrapper import Wrapper


@jax.tree_util.register_dataclass
@dataclass
class EpisodeStatistics:
    current_return: jax.Array
    episodic_return: jax.Array
    discounted_episodic_return: jax.Array
    length: jax.Array
    current_discounted_return: jax.Array
    current_length: jax.Array


class RecordEpisodeStatistics(Wrapper):
    """
    A wrapper that records episode statistics such as episodic return, discounted episodic return, and length.
    """

    gamma: float

    def __init__(self, env: Environment, gamma: float = 1.0):
        self.env = env
        self.gamma = gamma

    def init(self, key: jax.Array) -> EnvironmentState:
        env_state = self.env.init(key)
        episode_stats = EpisodeStatistics(
            current_return=jnp.float32(0.0),
            episodic_return=jnp.float32(0.0),
            current_discounted_return=jnp.float32(0.0),
            discounted_episodic_return=jnp.float32(0.0),
            length=jnp.int32(0.0),
            current_length=jnp.int32(0.0),
        )
        env_state = EnvironmentState(
            state=(env_state.state, episode_stats),
            obs=env_state.obs,
            true_obs=env_state.obs,
            reward=env_state.reward,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            info=env_state.info
            | {
                "current_return": episode_stats.current_return,
                "current_discounted_return": episode_stats.current_discounted_return,
                "current_length": episode_stats.current_length,
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
            true_obs=env_state.obs,
            reward=env_state.reward,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        env_state = self.env.reset(key, env_state)
        episode_stats = EpisodeStatistics(
            current_return=jnp.float32(0.0),
            current_length=jnp.int32(0.0),
            episodic_return=jnp.float32(0.0),
            discounted_episodic_return=jnp.float32(0.0),
            length=jnp.int32(0.0),
            current_discounted_return=jnp.float32(0.0),
        )
        env_state = EnvironmentState(
            state=(env_state, episode_stats),
            obs=env_state.obs,
            true_obs=env_state.obs,
            reward=env_state.reward,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            info=env_state.info
            | {
                "current_return": episode_stats.current_return,
                "current_discounted_return": episode_stats.current_discounted_return,
                "current_length": episode_stats.current_length,
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
            true_obs=env_state.obs,
            reward=env_state.reward,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        env_state = self.env.step(key, env_state, action)
        done = env_state.done
        reward = env_state.reward

        current_return = episode_stats._episodic_return + reward
        current_discounted_return = (
            episode_stats._discounted_episodic_return
            + reward * self.gamma**episode_stats._length
        )
        current_length = episode_stats.current_length + 1

        episodic_return = (
            1 - done
        ) * episode_stats.episodic_return + done * current_return
        discounted_episodic_return = (
            (1 - done) * episode_stats.discounted_episodic_return
            + done * current_discounted_return
        )
        length = (1 - done) * episode_stats.length + done * current_length

        current_return = (1 - done) * current_return
        current_discounted_return = (1 - done) * current_discounted_return
        current_length = (1 - done) * current_length

        episode_stats = EpisodeStatistics(
            current_return=current_return,
            current_discounted_return=current_discounted_return,
            current_length=current_length,
            episodic_return=episodic_return,
            discounted_episodic_return=discounted_episodic_return,
            length=length,
        )
        env_state = EnvironmentState(
            state=(env_state.state, episode_stats),
            obs=env_state.obs,
            true_obs=env_state.obs,
            reward=env_state.reward,
            terminated=env_state.terminated,
            truncated=env_state.truncated,
            info=env_state.info
            | {
                "current_return": episode_stats.current_return,
                "episodic_return": episode_stats.episodic_return,
                "discounted_episodic_return": episode_stats.discounted_episodic_return,
                "length": episode_stats.length,
            },
        )
        return env_state
