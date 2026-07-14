from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import EnvironmentWrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class EvaluateState(WrapperState):
    current_return: Array
    cumulative_return: Array
    n_episodes: Array

    @property
    def mean_return(self) -> Array:
        return self.cumulative_return / self.n_episodes


class Evaluate(EnvironmentWrapper[EvaluateState]):
    wrapped: Environment

    def __init__(self, wrapped: Environment):
        """
        Args:
            wrapped: The environment to wrap.
        """
        super().__init__(wrapped)

    def init(self, key: Key) -> tuple[EvaluateState, Timestep]:
        wrapped_state, timestep = self.wrapped.init(key)
        evaluate_state = EvaluateState(
            wrapped_state=wrapped_state,
            current_return=jax.numpy.zeros(timestep.reward.shape),
            cumulative_return=jax.numpy.zeros(timestep.reward.shape),
            n_episodes=jax.numpy.zeros(timestep.reward.shape),
        )
        return evaluate_state, timestep

    def reset(self, key: Key, state: EvaluateState) -> tuple[EvaluateState, Timestep]:
        evaluate_state = state
        wrapped_state, timestep = self.wrapped.reset(key, evaluate_state.wrapped_state)
        evaluate_state = EvaluateState(
            wrapped_state=wrapped_state,
            current_return=jax.numpy.zeros(timestep.reward.shape),
            cumulative_return=jax.numpy.zeros(timestep.reward.shape),
            n_episodes=jax.numpy.zeros(timestep.reward.shape),
        )
        return evaluate_state, timestep

    def step(
        self,
        key: Key,
        state: EvaluateState,
        action: PyTree,
    ) -> tuple[EvaluateState, Timestep]:
        evaluate_state = state
        wrapped_state, timestep = self.wrapped.step(
            key, evaluate_state.wrapped_state, action
        )
        current_return = evaluate_state.current_return + timestep.reward
        cumulative_return = jnp.where(
            timestep.done,
            evaluate_state.cumulative_return + current_return,
            evaluate_state.cumulative_return,
        )
        n_episodes = jnp.where(
            timestep.done,
            evaluate_state.n_episodes + 1,
            evaluate_state.n_episodes,
        )
        current_return = jnp.where(
            timestep.done,
            jnp.zeros_like(current_return),
            current_return,
        )
        evaluate_state = EvaluateState(
            wrapped_state=wrapped_state,
            current_return=current_return,
            cumulative_return=cumulative_return,
            n_episodes=n_episodes,
        )
        return evaluate_state, timestep
