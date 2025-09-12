import gymnax
import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep


class GymnaxEnvironment(Environment):
    """Base class for Gymnax environments."""

    env: gymnax.environments.environment.Environment
    env_params = gymnax.EnvParams

    def __init__(self, id: str, **kwargs):
        self.env, self.env_params = gymnax.make(id, **kwargs)

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        obs, gxm_state = self.env.reset(key, self.env_params)
        env_state = gxm_state
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=jnp.float32(0.0),
            terminated=jnp.bool(False),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(
        self, key: jax.Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: jax.Array, env_state: EnvironmentState, action: jax.Array
    ) -> tuple[EnvironmentState, Timestep]:
        gymnax_state = env_state
        obs, gymnax_state, reward, done, _ = self.env.step(
            key, gymnax_state, action, self.env_params
        )
        env_state = gymnax_state
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=reward,
            terminated=done,
            truncated=done,
            info={},
        )
        return env_state, timestep

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
