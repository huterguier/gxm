from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gxm.core import Environment, Timestep
from gxm.typing import Array, Key, PyTree
from gxm.wrappers.wrapper import Wrapper, WrapperState


@jax.tree_util.register_dataclass
@dataclass
class EvaluateState(WrapperState):
    current_return: Array
    cumulative_return: Array
    n_episodes: Array

    @property
    def mean_return(self) -> Array:
        return self.cumulative_return / self.n_episodes


class Evaluate(Wrapper[EvaluateState]):
    env: Environment

    def __init__(self, env: Environment, unwrap: bool = True):
        """
        Args:
            env: The environment to wrap.
            unwrap: Whether to unwrap the environment or treat it as part of the base environment.
        """
        super().__init__(env, unwrap=unwrap)

    def init(self, key: Key) -> tuple[EvaluateState, Timestep]:
        env_state, timestep = self.env.init(key)
        evaluate_state = EvaluateState(
            env_state=env_state,
            current_return=jax.numpy.zeros(timestep.reward.shape),
            cumulative_return=jax.numpy.zeros(timestep.reward.shape),
            n_episodes=jax.numpy.zeros(timestep.reward.shape),
        )
        return evaluate_state, timestep

    def reset(
        self, key: Key, env_state: EvaluateState
    ) -> tuple[EvaluateState, Timestep]:
        evaluate_state = env_state
        env_state, timestep = self.env.reset(key, evaluate_state.env_state)
        evaluate_state = EvaluateState(
            env_state=env_state,
            current_return=jax.numpy.zeros(timestep.reward.shape),
            cumulative_return=jax.numpy.zeros(timestep.reward.shape),
            n_episodes=jax.numpy.zeros(timestep.reward.shape),
        )
        return evaluate_state, timestep

    def step(
        self,
        key: Key,
        env_state: EvaluateState,
        action: PyTree,
    ) -> tuple[EvaluateState, Timestep]:
        evaluate_state = env_state
        env_state, timestep = self.env.step(key, evaluate_state.env_state, action)
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
            env_state=env_state,
            current_return=current_return,
            cumulative_return=cumulative_return,
            n_episodes=n_episodes,
        )
        return evaluate_state, timestep
