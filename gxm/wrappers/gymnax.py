import gymnax
import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState


class GymnaxEnvironment(Environment):
    """Base class for Gymnax environments."""

    env: gymnax.environments.environment.Environment
    env_params = gymnax.EnvParams

    def __init__(self, id: str, **kwargs):
        self.env, self.env_params = gymnax.make(id, **kwargs)

    def init(self, key: jax.Array) -> EnvironmentState:
        obs, state = self.env.reset(key, self.env_params)
        env_state = EnvironmentState(
            state=state, obs=obs, reward=jnp.float32(0.0), done=jnp.bool(False), info={}
        )
        return env_state

    def reset(self, key: jax.Array) -> EnvironmentState:
        obs, state = self.env.reset(key, self.env_params)
        env_state = EnvironmentState(
            state=state, obs=obs, reward=jnp.float32(0.0), done=jnp.bool(False), info={}
        )
        return env_state

    def step(
        self, key: jax.Array, env_state: EnvironmentState, action: jax.Array
    ) -> EnvironmentState:
        obs, state, reward, done, _ = self.env.step(
            key, env_state.state, action, self.env_params
        )
        env_state = EnvironmentState(
            state=state, obs=obs, reward=reward, done=done, info={}
        )
        return env_state

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
