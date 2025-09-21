from typing import Any

import jax
import jax.numpy as jnp
import jaxatari
import jaxatari.core
import jaxatari.environment
import jaxatari.spaces
import jaxatari.wrappers
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree


class JAXAtariEnvironment(Environment):
    """Base class for JAXAtari environments."""

    env: jaxatari.wrappers.JaxatariWrapper
    env_params: Any

    def __init__(self, id: str, **kwargs):
        env = jaxatari.core.make(id, **kwargs)
        env = AtariWrapper(env)
        self.env = PixelObsWrapper(env)
        self.action_space = self.jaxatari_to_gxm_space(self.env.action_space())

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        obs, jaxatari_state = self.env.reset(key)
        obs = self.to_grayscale(obs)
        obs = self.resize(obs)
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
        obs = self.to_grayscale(obs)
        obs = self.resize(obs)
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
    def to_grayscale(cls, obs: jax.Array) -> jax.Array:
        """Convert an RGB observation to grayscale."""
        return jnp.dot(obs[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))

    @classmethod
    def resize(cls, obs: jax.Array) -> jax.Array:
        """Resize an observation to 84x84."""
        return jax.image.resize(obs, (obs.shape[0], 84, 84), method="bilinear")

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
