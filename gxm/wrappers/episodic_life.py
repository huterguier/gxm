from dataclasses import dataclass

import jax
from jax import numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class EpisodicLifeState(WrapperState):
    lives: Array


class EpisodicLife(Wrapper[EpisodicLifeState]):
    """
    A wrapper that makes losing a life in an environment (like Atari games) count as the end of an episode.
    It assumes that the environment's timestep info dictionary contains a "lives" key indicating the number of lives remaining.
    """

    env: Environment

    def __init__(self, env: Environment):
        """
        Args:
            env: The environment to wrap.
        """
        self.env = env

    def init(self, key: Key) -> tuple[EpisodicLifeState, Timestep]:
        env_state, timestep = self.env.init(key)
        lives = timestep.info["lives"]
        episodic_life_state = EpisodicLifeState(env_state=env_state, lives=lives)
        return episodic_life_state, timestep

    def reset(
        self, key: Key, env_state: EpisodicLifeState
    ) -> tuple[EpisodicLifeState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state.env_state)
        lives = timestep.info["lives"]
        episodic_life_state = EpisodicLifeState(env_state=env_state, lives=lives)
        return episodic_life_state, timestep

    def step(
        self,
        key: Key,
        env_state: EpisodicLifeState,
        action: PyTree,
    ) -> tuple[EpisodicLifeState, Timestep]:
        prev_lives = env_state.lives
        env_state, timestep = self.env.step(key, env_state.env_state, action)
        lives = timestep.info["lives"]
        episodic_life_state = EpisodicLifeState(env_state=env_state, lives=lives)
        timestep.terminated = jnp.logical_or(timestep.terminated, lives < prev_lives)
        return episodic_life_state, timestep
