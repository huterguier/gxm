from dataclasses import dataclass

import gymnax
import jax
import jax.numpy as jnp

from gxm.core import Env, EnvState, State


@jax.tree_util.register_dataclass
@dataclass
class GymnaxState(State):
    state_gymnax: gymnax.EnvState


class GymnaxEnv(Env[GymnaxState]):
    """Base class for Gymnax environments."""

    env: gymnax.environments.environment.Environment
    env_params = gymnax.EnvParams

    def __init__(self, env_id: str, **kwargs):
        self.env, self.env_params = gymnax.make(env_id, **kwargs)

    def step(
        self, key: jax.Array, state: EnvState | GymnaxState, action: jax.Array
    ) -> EnvState:
        state = state.state if isinstance(state, EnvState) else state
        obs, state_gymnax, reward, done, _ = self.env.step(
            key, state.state_gymnax, action, self.env_params
        )
        state = GymnaxState(time=state.time + 1, state_gymnax=state_gymnax)
        env_state = EnvState(state=state, obs=obs, reward=reward, done=done, info={})
        return env_state

    def reset(self, key: jax.Array) -> EnvState:
        obs, state_gymnax = self.env.reset(key)
        state = GymnaxState(time=0, state_gymnax=state_gymnax)
        env_state = EnvState(
            state=state, obs=obs, reward=jnp.float32(0.0), done=jnp.bool(False), info={}
        )
        return env_state

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
