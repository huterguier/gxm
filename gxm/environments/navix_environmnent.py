import jax
import jax.numpy as jnp
import navix

from gxm.core import Environment, EnvironmentState, Timestep


class NavixEnvironment(Environment):
    """Base class for Gymnax environments."""

    env: navix.Environment

    def __init__(self, id: str, **kwargs):
        self.env, self.env_params = navix.make(id, **kwargs)

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        navix_state = self.env.reset(key)
        env_state = navix_state
        timestep = Timestep(
            obs=navix_state.observation,
            true_obs=navix_state.observation,
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
        del key
        gymnax_state = env_state
        navix_state = self.env.step(gymnax_state, action)
        env_state = gymnax_state
        timestep = Timestep(
            obs=navix_state.observation,
            true_obs=navix_state.observation,
            reward=navix_state.reward,
            terminated=navix_state.is_done(),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    @property
    def num_actions(self) -> int:
        return len(self.env.action_set)
