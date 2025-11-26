from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class EpisodeCounterState(WrapperState):
    n_episodes: Array


class EpisodeCounter(Wrapper):
    """A wrapper that counts the number of episodes completed in the environment."""

    def __init__(self, env: Environment, unwrap: bool = True):
        super().__init__(env, unwrap=unwrap)

    def init(self, key: Key) -> tuple[EpisodeCounterState, Timestep]:
        env_state, timestep = self.env.init(key)
        episode_counter_state = EpisodeCounterState(
            env_state=env_state,
            n_episodes=jnp.int32(0),
        )
        timestep.info["n_episodes"] = episode_counter_state.n_episodes
        return episode_counter_state, timestep

    def reset(
        self, key: Key, env_state: EpisodeCounterState
    ) -> tuple[EpisodeCounterState, Timestep]:
        episode_counter_state = env_state
        env_state, timestep = self.env.reset(key, episode_counter_state.env_state)
        episode_counter_state = EpisodeCounterState(
            env_state=env_state,
            n_episodes=episode_counter_state.n_episodes,
        )
        timestep.info["n_episodes"] = episode_counter_state.n_episodes
        return episode_counter_state, timestep

    def step(
        self,
        key: Key,
        env_state: EpisodeCounterState,
        action: PyTree,
    ) -> tuple[EpisodeCounterState, Timestep]:
        episode_counter_state = env_state
        env_state, timestep = self.env.step(key, env_state.env_state, action)
        episode_counter_state = EpisodeCounterState(
            env_state=env_state,
            n_episodes=jnp.where(
                timestep.done,
                episode_counter_state.n_episodes + 1,
                episode_counter_state.n_episodes,
            ),
        )
        timestep.info["n_episodes"] = episode_counter_state.n_episodes
        return episode_counter_state, timestep
