import jax
import gymnax
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeVar,
)


TEnvState = TypeVar("TEnvState", bound="EnvState")

@jax.tree_util.register_dataclass
@dataclass
class EnvState:
    time: int


class Env(Generic[TEnvState]):
    """Base class for environments in gxm."""

    def step(
        self,
        key: jax.Array,
        state: tuple | TEnvState,
        action: jax.Array,
    ) -> tuple[TEnvState, Any, jax.Array, jax.Array, Any]:
        """Perform a step in the environment given an action."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self, key: jax.Array) -> tuple[TEnvState, Any, jax.Array, jax.Array, Any]:
        """Reset the environment to its initial state."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def num_actions(self) -> int:
        """Return the number of actions available in the environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")


@jax.tree_util.register_dataclass
@dataclass
class GymnaxEnvState(EnvState):
    env_state: gymnax.EnvState


class GymnaxEnv(Env[GymnaxEnvState]):
    """Base class for Gymnax environments."""
    env: gymnax.environments.environment.Environment
    env_params = gymnax.EnvParams

    def __init__(self, env_id: str, **kwargs):
        self.env, self.env_params = gymnax.make(env_id, **kwargs)

    def step(
        self, 
        key: jax.Array, 
        state: tuple | GymnaxEnvState, 
        action: jax.Array
    ) -> tuple[GymnaxEnvState, Any, jax.Array, jax.Array, Any]:
        state = state[0] if isinstance(state, tuple) else state
        obs, env_state, reward, done, info = self.env.step(key, state.env_state, action)
        state = GymnaxEnvState(time=state.time+1, env_state=env_state)
        return state, obs, reward, done, info

    def reset(self, key: jax.Array):
        obs, env_state = self.env.reset(key)
        state = GymnaxEnvState(time=0, env_state=env_state)
        return state, obs, jax.numpy.zeros((1,), dtype=jax.numpy.float32), jax.numpy.zeros((1,), dtype=jax.numpy.bool_), {}

    @property
    def num_actions(self) -> int:
        return self.env.num_actions


def make(env_id, **kwargs):
    return GymnaxEnv(env_id, **kwargs)


if __name__ == "__main__":

    def rollout1(env, key, num_steps):
        state, obs, reward, done, info = env.reset(key)
        for _ in range(num_steps):
            action = jax.random.randint(key, (1,), 0, env.num_actions)[0]
            state, obs, reward, done, info = env.step(key, state, action)
            print(f"Step: {state.time}, Action: {action}, Reward: {reward}, Done: {done}")
            if done:
                break

    def rollout2(env, key, num_steps):
        state = env.reset(key)
        for _ in range(num_steps):
            action = jax.random.randint(key, (1,), 0, env.num_actions)[0]
            state = env.step(key, state, action)
            _, obs, reward, done, info = state
            print(f"Step: {state[0].time}, Action: {action}, Reward: {reward}, Done: {done}")
            if done:
                break

    env = make("CartPole-v1")
    key = jax.random.PRNGKey(0)
    num_steps = 100
    rollout1(env, key, num_steps)
    rollout2(env, key, num_steps)
