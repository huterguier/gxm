import jax
import gymnax
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeVar,
)


TState = TypeVar("TState", bound="State")

@jax.tree_util.register_dataclass
@dataclass
class State:
    time: int


@jax.tree_util.register_dataclass
@dataclass
class EnvState:
    state: State
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Any

    def __getitem__(self, item):
        return (self.state, self.obs, self.reward, self.done, self.info)[item]

    def __iter__(self):
        return iter((self.state, self.obs, self.reward, self.done, self.info))



class Env(Generic[TState]):
    """Base class for environments in gxm."""

    def step(
        self,
        key: jax.Array,
        state: tuple | TState,
        action: jax.Array,
    ) -> EnvState:
        """Perform a step in the environment given an action."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self, key: jax.Array) -> EnvState:
        """Reset the environment to its initial state."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def num_actions(self) -> int:
        """Return the number of actions available in the environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")


@jax.tree_util.register_dataclass
@dataclass
class GymnaxState(State):
    env_state: gymnax.EnvState


class GymnaxEnv(Env[GymnaxState]):
    """Base class for Gymnax environments."""
    env: gymnax.environments.environment.Environment
    env_params = gymnax.EnvParams

    def __init__(self, env_id: str, **kwargs):
        self.env, self.env_params = gymnax.make(env_id, **kwargs)

    def step(
        self, 
        key: jax.Array, 
        state: EnvState | GymnaxState, 
        action: jax.Array
    ) -> EnvState:
        state = state.state if isinstance(state, EnvState) else state
        obs, env_state, reward, done, info = self.env.step(key, state.env_state, action)
        state = GymnaxState(time=state.time+1, env_state=env_state)
        env_state = EnvState(
            state=state,
            obs=obs,
            reward=jax.numpy.array([reward], dtype=jax.numpy.float32),
            done=jax.numpy.array([done], dtype=jax.numpy.bool_),
            info=info
        )
        return env_state

    def reset(self, key: jax.Array) -> EnvState:
        obs, env_state = self.env.reset(key)
        state = GymnaxState(time=0, env_state=env_state)
        env_state = EnvState(
            state=state,
            obs=obs,
            reward=jax.numpy.zeros((1,), dtype=jax.numpy.float32),
            done=jax.numpy.zeros((1,), dtype=jax.numpy.bool_),
            info={}
        )
        return env_state

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
        env_state = env.reset(key)
        for _ in range(num_steps):
            action = jax.random.randint(key, (1,), 0, env.num_actions)[0]
            env_state = env.step(key, env_state, action)
            _, obs, reward, done, info = env_state
            print(f"Step: {env_state[0].time}, Action: {action}, Reward: {reward}, Done: {done}")
            if done:
                break

    env = make("CartPole-v1")
    key = jax.random.PRNGKey(0)
    num_steps = 100
    rollout1(env, key, num_steps)
    rollout2(env, key, num_steps)
