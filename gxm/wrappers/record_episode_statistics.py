from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class EpisodeStatistics:
    episode_length: Array
    episodic_return: Array
    episodic_discounted_return: Array
    mask: Array


@jax.tree_util.register_dataclass
@dataclass
class CurrentStatistics:
    current_return: Array
    current_length: Array
    current_discounted_return: Array


@jax.tree_util.register_dataclass
@dataclass
class RecordEpisodeStatisticsState(WrapperState):
    current_stats: CurrentStatistics
    episode_stats: EpisodeStatistics

    def __iter__(self):
        yield self.env_state
        yield self.current_stats
        yield self.episode_stats


class RecordEpisodeStatistics(Wrapper[RecordEpisodeStatisticsState]):
    """
    A wrapper that records the episode length :math:`T` , episodic return
    :math:`J(\\tau) = \\sum_{t=0}^{T} r_t` , and discounted episodic return
    :math:`G(\\tau) = \\sum_{t=0}^{T} \\gamma^t r_t` at the end of each episode.
    The statistics can be accessed from the ``info`` field of the ``Timestep`` returned
    by the environment. It will contain the stats of the most recent finished episode.
    By default , the discount factor :math:`\\gamma` is set to 1.0, meaning that the
    episodic return and discounted episodic return are the same.
    """

    gamma: float
    """The discount factor :math:`\\gamma` for calculating the discounted episodic return."""
    n_episodes: int
    """The number of past episodes to record statistics for."""

    def __init__(
        self,
        env: Environment,
        unwrap: bool = True,
        gamma: float = 1.0,
        n_episodes: int = 1,
    ):
        super().__init__(env, unwrap=unwrap)
        self.gamma = gamma
        self.n_episodes = n_episodes

    @staticmethod
    def get_averaged_stats(episode_stats: EpisodeStatistics) -> dict[str, jax.Array]:
        valid_episodes = jnp.maximum(jnp.sum(episode_stats.mask), 1.0)
        episode_length = (
            jnp.sum(episode_stats.episode_length * episode_stats.mask) / valid_episodes
        )
        episodic_return = (
            jnp.sum(episode_stats.episodic_return * episode_stats.mask) / valid_episodes
        )
        episodic_discounted_return = (
            jnp.sum(episode_stats.episodic_discounted_return * episode_stats.mask)
            / valid_episodes
        )
        return {
            "episode_length": episode_length,
            "episodic_return": episodic_return,
            "episodic_discounted_return": episodic_discounted_return,
        }

    def init(self, key: Key) -> tuple[RecordEpisodeStatisticsState, Timestep]:
        env_state, timestep = self.env.init(key)
        current_stats = CurrentStatistics(
            current_return=jnp.float32(0.0),
            current_length=jnp.int32(0.0),
            current_discounted_return=jnp.float32(0.0),
        )
        episode_stats = EpisodeStatistics(
            episodic_return=jnp.zeros(self.n_episodes, dtype=jnp.float32),
            episodic_discounted_return=jnp.zeros(self.n_episodes, dtype=jnp.float32),
            episode_length=jnp.zeros(self.n_episodes, dtype=jnp.int32),
            mask=jnp.zeros(self.n_episodes, dtype=jnp.int32),
        )
        record_episode_stats_state = RecordEpisodeStatisticsState(
            env_state=env_state,
            current_stats=current_stats,
            episode_stats=episode_stats,
        )
        timestep.info |= {
            "current_length": current_stats.current_length,
            "current_return": current_stats.current_return,
            "current_discounted_return": current_stats.current_discounted_return,
        }
        timestep.info |= self.get_averaged_stats(episode_stats)
        return record_episode_stats_state, timestep

    def reset(
        self, key: Key, env_state: RecordEpisodeStatisticsState
    ) -> tuple[RecordEpisodeStatisticsState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state.env_state)
        current_stats = CurrentStatistics(
            current_return=jnp.float32(0.0),
            current_length=jnp.int32(0.0),
            current_discounted_return=jnp.float32(0.0),
        )
        episode_stats = EpisodeStatistics(
            episodic_return=jnp.zeros(self.n_episodes, dtype=jnp.float32),
            episodic_discounted_return=jnp.zeros(self.n_episodes, dtype=jnp.float32),
            episode_length=jnp.zeros(self.n_episodes, dtype=jnp.int32),
            mask=jnp.zeros(self.n_episodes, dtype=jnp.int32),
        )
        record_episode_stats_state = RecordEpisodeStatisticsState(
            env_state=env_state,
            current_stats=current_stats,
            episode_stats=episode_stats,
        )
        timestep.info |= {
            "current_length": current_stats.current_length,
            "episode_length": episode_stats.episode_length,
            "current_return": current_stats.current_return,
        }
        timestep.info |= self.get_averaged_stats(episode_stats)
        return record_episode_stats_state, timestep

    def step(
        self,
        key: Key,
        env_state: RecordEpisodeStatisticsState,
        action: PyTree,
    ) -> tuple[RecordEpisodeStatisticsState, Timestep]:
        wrapper_state = env_state
        current_stats = wrapper_state.current_stats
        episode_stats = wrapper_state.episode_stats

        env_state, timestep = self.env.step(key, wrapper_state.env_state, action)

        done = timestep.done
        reward = timestep.reward

        current_return = current_stats.current_return + reward
        current_discounted_return = (
            current_stats.current_discounted_return
            + reward * self.gamma**current_stats.current_length
        )
        current_length = current_stats.current_length + 1

        episodic_return = jnp.where(
            done,
            jnp.roll(episode_stats.episodic_return, 1).at[0].set(current_return),
            episode_stats.episodic_return,
        )
        episodic_discounted_return = jnp.where(
            done,
            jnp.roll(episode_stats.episodic_discounted_return, 1)
            .at[0]
            .set(current_discounted_return),
            episode_stats.episodic_discounted_return,
        )
        episode_length = jnp.where(
            done,
            jnp.roll(episode_stats.episode_length, 1).at[0].set(current_length),
            episode_stats.episode_length,
        )
        mask = jnp.where(
            done,
            jnp.roll(episode_stats.mask, 1).at[0].set(1),
            episode_stats.mask,
        )

        current_return = (1 - done) * current_return
        current_discounted_return = (1 - done) * current_discounted_return
        current_length = (1 - done) * current_length

        current_stats = CurrentStatistics(
            current_return=current_return,
            current_length=current_length,
            current_discounted_return=current_discounted_return,
        )
        episode_stats = EpisodeStatistics(
            episode_length=episode_length,
            episodic_return=episodic_return,
            episodic_discounted_return=episodic_discounted_return,
            mask=mask,
        )
        wrapper_state = RecordEpisodeStatisticsState(
            env_state=env_state,
            current_stats=current_stats,
            episode_stats=episode_stats,
        )
        timestep.info |= {
            "current_length": current_stats.current_length,
            "current_return": current_stats.current_return,
            "current_discounted_return": current_stats.current_discounted_return,
        }
        timestep.info |= self.get_averaged_stats(episode_stats)

        return wrapper_state, timestep
