import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.typing import Array
from gxm.wrappers.wrapper import Wrapper


@jax.tree_util.register_dataclass
@dataclass
class EpisodeStatistics:
    current_return: Array
    episodic_return: Array
    current_length: Array
    episode_length: Array
    current_discounted_return: Array
    episodic_discounted_return: Array


class RecordEpisodeStatistics(Wrapper):
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

    def __init__(self, env: Environment, gamma: float = 1.0):
        self.env = env
        self.gamma = gamma

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.init(key)
        episode_stats = EpisodeStatistics(
            current_return=jnp.float32(0.0),
            episodic_return=jnp.float32(0.0),
            current_discounted_return=jnp.float32(0.0),
            episodic_discounted_return=jnp.float32(0.0),
            episode_length=jnp.int32(0.0),
            current_length=jnp.int32(0.0),
        )
        env_state = (env_state, episode_stats)
        timestep.info |= {
            "current_length": episode_stats.current_length,
            "episode_length": episode_stats.episode_length,
            "current_return": episode_stats.current_return,
            "episodic_return": episode_stats.episodic_return,
            "current_discounted_return": episode_stats.current_discounted_return,
            "episodic_discounted_return": episode_stats.episodic_discounted_return,
        }
        return env_state, timestep

    def reset(
        self, key: jax.Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        (env_state, episode_stats) = env_state
        env_state, timestep = self.env.reset(key, env_state)
        episode_stats = EpisodeStatistics(
            current_return=jnp.float32(0.0),
            current_length=jnp.int32(0.0),
            episodic_return=jnp.float32(0.0),
            episodic_discounted_return=jnp.float32(0.0),
            episode_length=jnp.int32(0.0),
            current_discounted_return=jnp.float32(0.0),
        )
        env_state = (env_state, episode_stats)
        timestep.info |= {
            "current_length": episode_stats.current_length,
            "episode_length": episode_stats.episode_length,
            "current_return": episode_stats.current_return,
            "episodic_return": episode_stats.episodic_return,
            "current_discounted_return": episode_stats.current_discounted_return,
            "episodic_discounted_return": episode_stats.episodic_discounted_return,
        }
        return env_state, timestep

    def step(
        self,
        key: jax.Array,
        env_state: EnvironmentState,
        action: jax.Array,
    ) -> tuple[EnvironmentState, Timestep]:
        (env_state, episode_stats) = env_state
        env_state, timestep = self.env.step(key, env_state, action)

        done = timestep.done
        reward = timestep.reward

        current_return = episode_stats.current_return + reward
        current_discounted_return = (
            episode_stats.current_discounted_return
            + reward * self.gamma**episode_stats.current_length
        )
        current_length = episode_stats.current_length + 1

        episodic_return = (
            1 - done
        ) * episode_stats.episodic_return + done * current_return
        episodic_discounted_return = (
            (1 - done) * episode_stats.episodic_discounted_return
            + done * current_discounted_return
        )
        episode_length = (
            1 - done
        ) * episode_stats.episode_length + done * current_length

        current_return = (1 - done) * current_return
        current_discounted_return = (1 - done) * current_discounted_return
        current_length = (1 - done) * current_length

        episode_stats = EpisodeStatistics(
            current_length=current_length,
            episode_length=episode_length,
            current_return=current_return,
            episodic_return=episodic_return,
            current_discounted_return=current_discounted_return,
            episodic_discounted_return=episodic_discounted_return,
        )
        env_state = (env_state, episode_stats)
        timestep.info |= {
            "current_length": episode_stats.current_length,
            "episode_length": episode_stats.episode_length,
            "current_return": episode_stats.current_return,
            "episodic_return": episode_stats.episodic_return,
            "current_discounted_return": episode_stats.current_discounted_return,
            "episodic_discounted_return": episode_stats.episodic_discounted_return,
        }

        return env_state, timestep
