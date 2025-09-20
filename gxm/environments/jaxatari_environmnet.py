from typing import Any

import gymnax.environments.spaces
import jax
import jax.numpy as jnp
import jaxatari
import jaxatari.core
import jaxatari.environment
import jaxatari.spaces

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree


class JAXAtariEnvironment(Environment):
    """Base class for JAXAtari environments."""

    env: jaxatari.environment.JaxEnvironment
    env_params: Any

    def __init__(self, id: str, **kwargs):
        self.env = jaxatari.core.make(id, **kwargs)
        self.action_space = self.jaxatari_to_gxm_space(self.env.action_space())

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        obs, jaxatari_state = self.env.reset(key)
        env_state = jaxatari_state
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
        obs, gymnax_state, reward, done, _ = self.env.step(gymnax_state, action)
        env_state = gymnax_state
        timestep = Timestep(
            obs=obs,
            true_obs=obs,
            reward=jnp.float32(reward),
            terminated=jnp.bool(done),
            truncated=jnp.bool(done),
            info={},
        )
        return env_state, timestep

    @classmethod
    def jaxatari_to_gxm_space(cls, jaxatari_space) -> Space:
        """Convert a Gymnax space to a Gxm space."""
        if isinstance(jaxatari_space, jaxatari.spaces.Discrete):
            return Discrete(jaxatari_space.n)
        if isinstance(jaxatari_space, jaxatari.spaces.Box):
            return Box(
                low=jaxatari_space.low,
                high=jaxatari_space.high,
                shape=jaxatari_space.shape,
            )
        if isinstance(jaxatari_space, jaxatari.spaces.Dict):
            return Tree(
                {
                    k: cls.jaxatari_to_gxm_space(v)
                    for k, v in jaxatari_space.spaces.items()
                }
            )
        else:
            raise NotImplementedError(
                f"JAXAtari space type {type(jaxatari_space)} not supported."
            )
