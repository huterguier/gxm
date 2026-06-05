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
class JAXAtariState(EnvironmentState):
    jaxatari_state: Any


class JAXAtariAdapter(Environment[JAXAtariState]):
    jaxatari_id: str
    env: jaxatari.wrappers.JaxatariWrapper

    def __init__(self, id: str, **kwargs):
        self.jaxatari_id = id
        self.id = f"JAXAtari/{id}"
        env = jaxatari.core.make(id, **kwargs)
        env = AtariWrapper(env)
        self.env = PixelObsWrapper(env)
        self.action_space = _jaxatari_to_gxm_space(self.env.action_space())

    def init(self, key: Key) -> tuple[JAXAtariState, Timestep]:
        obs, jaxatari_state = self.env.reset(key)
        obs = _to_grayscale(obs)
        obs = _resize(obs)
        env_state = JAXAtariState(jaxatari_state=jaxatari_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=self.action_space.sample(key),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(False),
            truncated=jnp.bool(False),
            info={},
        )
        return env_state, timestep

    def reset(self, key: Key, env_state: JAXAtariState) -> tuple[JAXAtariState, Timestep]:
        del env_state
        return self.init(key)

    def step(self, key: Key, env_state: JAXAtariState, action: Array) -> tuple[JAXAtariState, Timestep]:
        del key
        obs, jaxatari_state, reward, done, _ = self.env.step(env_state.jaxatari_state, action)
        obs = _to_grayscale(obs)
        obs = _resize(obs)
        env_state = JAXAtariState(jaxatari_state=jaxatari_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=action,
            reward=jnp.float32(reward),
            terminated=jnp.bool(done),
            truncated=jnp.bool(done),
            info={},
        )
        return env_state, timestep


def _to_grayscale(obs: Array) -> Array:
    return jnp.dot(obs[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))


def _resize(obs: Array) -> Array:
    return jax.image.resize(obs, (obs.shape[0], 84, 84), method="bilinear")


def _jaxatari_to_gxm_space(jaxatari_space) -> Space:
    if isinstance(jaxatari_space, jaxatari.spaces.Discrete):
        return Discrete(jaxatari_space.n)
    if isinstance(jaxatari_space, jaxatari.spaces.Box):
        return Box(low=jaxatari_space.low, high=jaxatari_space.high, shape=jaxatari_space.shape)
    if isinstance(jaxatari_space, jaxatari.spaces.Dict):
        return Tree({k: _jaxatari_to_gxm_space(v) for k, v in jaxatari_space.spaces.items()})
    raise NotImplementedError(f"JAXAtari space type {type(jaxatari_space)} not supported.")


def make(id: str, **kwargs) -> Environment:
    return JAXAtariAdapter(id, **kwargs)
