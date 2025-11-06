from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jaxatari
import jaxatari.core
import jaxatari.spaces
import jaxatari.wrappers
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key


@jax.tree_util.register_dataclass
@dataclass
class JAXAtariEnvironmentState(EnvironmentState):
    """State for JAXAtari environments."""

    jaxatari_state: Any
    """State for JAXAtari environments."""


class JAXAtariEnvironment(Environment[JAXAtariEnvironmentState]):
    """Base class for JAXAtari environments."""

    jaxatari_id: str
    """The JAXAtari environment id."""
    env: jaxatari.wrappers.JaxatariWrapper
    """The JAXAtari environment."""
    env_params: Any
    """The JAXAtari environment parameters."""

    def __init__(self, id: str, **kwargs):

        self.id = id
        self.jaxatari_id = id.split("/", 1)[1]
        env = jaxatari.core.make(self.jaxatari_id, **kwargs)
        env = AtariWrapper(env)
        self.env = PixelObsWrapper(env)
        self.action_space = self.jaxatari_to_gxm_space(self.env.action_space())

    def init(self, key: Key) -> tuple[JAXAtariEnvironmentState, Timestep]:
        obs, jaxatari_state = self.env.reset(key)
        obs = self.to_grayscale(obs)
        obs = self.resize(obs)
        env_state = JAXAtariEnvironmentState(jaxatari_state=jaxatari_state)
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
        self, key: Key, env_state: JAXAtariEnvironmentState
    ) -> tuple[JAXAtariEnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: Key, env_state: JAXAtariEnvironmentState, action: Array
    ) -> tuple[JAXAtariEnvironmentState, Timestep]:
        del key
        jaxatari_state = env_state.jaxatari_state
        obs, jaxatari_state, reward, done, _ = self.env.step(jaxatari_state, action)
        obs = self.to_grayscale(obs)
        obs = self.resize(obs)
        env_state = JAXAtariEnvironmentState(jaxatari_state=jaxatari_state)
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
    def to_grayscale(cls, obs: Array) -> Array:
        """Convert an RGB observation to grayscale."""
        return jnp.dot(obs[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))

    @classmethod
    def resize(cls, obs: Array) -> Array:
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
