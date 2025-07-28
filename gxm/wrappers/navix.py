from dataclasses import dataclass

import jax
import jax.numpy as jnp
import navix

from gxm.core import Env, EnvState, State


@jax.tree_util.register_dataclass
@dataclass
class NavixState(State):
    state_navix: navix.Timestep


class NavixEnv(Env[NavixState]):
    """Base class for Gymnax environments."""

    env: navix.environments.Environment

    def __init__(self, env_id: str, **kwargs):
        self.env, self.env_params = navix.make(env_id, **kwargs)

    def step(
        self, key: jax.Array, state: EnvState | NavixState, action: jax.Array
    ) -> EnvState:
        state = state.state if isinstance(state, EnvState) else state
        state_navix = self.env.step(state.state_navix, action)
        state = NavixState(time=state.time + 1, state_navix=state_navix)
        env_state = EnvState(
            state=state,
            obs=state_navix.observation,
            reward=timestep.reward,
            done=timestep.done,
            info={},
        )
        return env_state

    def reset(self, key: jax.Array) -> EnvState:
        timestep = self.env.reset(key)
        state = NavixState(time=0, state_navix=timestep.state)
        env_state = EnvState(
            state=state, obs=obs, reward=jnp.float32(0.0), done=jnp.bool(False), info={}
        )
        return env_state

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
