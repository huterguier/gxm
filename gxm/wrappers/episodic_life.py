import dataclasses
from dataclasses import dataclass

import jax
from jax import numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import EnvironmentWrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class EpisodicLifeState(WrapperState):
    lives: Array


class EpisodicLife(EnvironmentWrapper[EpisodicLifeState]):
    """
    A wrapper that makes losing a life in an environment (like Atari games) count as the end of an episode.
    It assumes that the environment's timestep info dictionary contains a "lives" key indicating the number of lives remaining.
    """

    wrapped: Environment

    def __init__(self, wrapped: Environment):
        """
        Args:
            wrapped: The environment to wrap.
        """
        super().__init__(wrapped)

    def init(self, key: Key) -> tuple[EpisodicLifeState, Timestep]:
        wrapped_state, timestep = self.wrapped.init(key)
        lives = timestep.info["lives"]
        episodic_life_state = EpisodicLifeState(
            wrapped_state=wrapped_state, lives=lives
        )
        return episodic_life_state, timestep

    def reset(
        self, key: Key, state: EpisodicLifeState
    ) -> tuple[EpisodicLifeState, Timestep]:
        wrapped_state, timestep = self.wrapped.reset(key, state.wrapped_state)
        lives = timestep.info["lives"]
        episodic_life_state = EpisodicLifeState(
            wrapped_state=wrapped_state, lives=lives
        )
        return episodic_life_state, timestep

    def step(
        self,
        key: Key,
        state: EpisodicLifeState,
        action: PyTree,
    ) -> tuple[EpisodicLifeState, Timestep]:
        prev_lives = state.lives
        wrapped_state, timestep = self.wrapped.step(key, state.wrapped_state, action)
        lives = timestep.info["lives"]
        episodic_life_state = EpisodicLifeState(
            wrapped_state=wrapped_state, lives=lives
        )
        timestep = dataclasses.replace(
            timestep, terminated=jnp.logical_or(timestep.terminated, lives < prev_lives)
        )
        return episodic_life_state, timestep
