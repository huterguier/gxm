from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
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
    :math:`J(\\tau) = \\sum_{t=0}^{T} r_t` and discounted return :math:`G(\\tau) = \\sum_{t=0}^{T} \\gamma^t r_t`
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
            discounted_episodic_return=jnp.float32(0.0),
            length=jnp.int32(0.0),
            current_length=jnp.int32(0.0),
        )
        env_state = (env_state, episode_stats)
        timestep.info |= {
            "current_return": episode_stats.current_return,
            "current_discounted_return": episode_stats.current_discounted_return,
            "current_length": episode_stats.current_length,
            "episodic_return": episode_stats.episodic_return,
            "discounted_episodic_return": episode_stats.discounted_episodic_return,
            "length": episode_stats.length,
        }
        return env_state, timestep

    def reset(
        self, key: jax.Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        (env_state, episode_stats) = env_state
        env_state = self.env.reset(key, env_state)
        episode_stats = EpisodeStatistics(
            current_return=jnp.float32(0.0),
            current_length=jnp.int32(0.0),
            episodic_return=jnp.float32(0.0),
            discounted_episodic_return=jnp.float32(0.0),
            length=jnp.int32(0.0),
            current_discounted_return=jnp.float32(0.0),
        )
        env_state = (env_state, episode_stats)
        timestep = Timestep(
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
        env_state = (env_state, episode_stats)
        timestep.info |= {
            "current_return": episode_stats.current_return,
            "current_discounted_return": episode_stats.current_discounted_return,
            "current_length": episode_stats.current_length,
            "episodic_return": episode_stats.episodic_return,
            "discounted_episodic_return": episode_stats.discounted_episodic_return,
            "length": episode_stats.length,
        }

        return env_state, timestep
