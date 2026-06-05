from dataclasses import dataclass
from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key

_envs_gymnasium = {}


@jax.tree_util.register_dataclass
@dataclass
class GymnasiumState(EnvironmentState):
    env_id: Array


class GymnasiumAdapter(Environment[GymnasiumState]):
    gymnasium_id: str
    return_shape_dtype: Any
    kwargs: Any

    def __init__(self, id: str, **kwargs):
        self.gymnasium_id = id
        self.id = f"Gymnasium/{id}"
        env = gymnasium.make_vec(id, num_envs=1, **kwargs)
        obs, info_reset = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info_step = env.step(action)
        if jax.tree.structure(info_reset) != jax.tree.structure(info_step):
            info = info_step
        else:
            info = info_reset
        env_state = GymnasiumState(env_id=jnp.int32(0))
        timestep = Timestep(
            next_obs=jnp.array(obs),
            true_next_obs=jnp.array(obs),
            action=jnp.array(action),
            reward=jnp.array(reward, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool),
            truncated=jnp.array(truncated, dtype=jnp.bool),
            info=jax.tree.map(jnp.array, info),
        )
        self.return_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), (env_state, timestep)
        )
        self.action_space = _gymnasium_to_gxm_space(env.single_action_space)
        self.observation_space = _gymnasium_to_gxm_space(env.single_observation_space)
        self.kwargs = kwargs

    def init(self, key: Key) -> tuple[GymnasiumState, Timestep]:
        def callback(key):
            global _envs_gymnasium
            shape = key.shape[:-1]
            keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]
            envs = gymnasium.make_vec(self.gymnasium_id, num_envs=num_envs, **self.kwargs)
            obs, info = envs.reset(seed=0)
            if jax.tree.structure(info) != jax.tree.structure(self.return_shape_dtype[1].info):
                info = jax.tree.map(
                    lambda x: jnp.zeros((num_envs,) + x.shape[1:], x.dtype),
                    self.return_shape_dtype[1].info,
                )
            env_id = len(_envs_gymnasium)
            _envs_gymnasium[env_id] = envs
            action_spec = self.return_shape_dtype[1].action
            action_sentinel = jnp.zeros((num_envs,) + action_spec.shape, action_spec.dtype)
            env_state = GymnasiumState(env_id=jnp.full(shape, env_id, dtype=jnp.int32))
            timestep = Timestep(
                next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                action=jnp.reshape(action_sentinel, shape + action_spec.shape),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.zeros(shape, dtype=jnp.bool),
                truncated=jnp.zeros(shape, dtype=jnp.bool),
                info=jax.tree.map(lambda i: jnp.reshape(i, shape + i.shape[1:]), info),
            )
            return env_state, timestep

        env_state, timestep = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            jax.random.key_data(key),
            vmap_method="broadcast_all",
        )
        return env_state, timestep

    def reset(self, key: Key, env_state: GymnasiumState) -> tuple[GymnasiumState, Timestep]:
        del key

        def callback(env_id):
            global _envs_gymnasium
            shape = env_id.shape
            envs = _envs_gymnasium[np.ravel(env_id)[0]]
            num_envs = env_id.size
            obs, info = envs.reset()
            if jax.tree.structure(info) != jax.tree.structure(self.return_shape_dtype[1].info):
                info = jax.tree.map(
                    lambda x: jnp.zeros((num_envs,) + x.shape[1:], x.dtype),
                    self.return_shape_dtype[1].info,
                )
            action_spec = self.return_shape_dtype[1].action
            action_sentinel = jnp.zeros((num_envs,) + action_spec.shape, action_spec.dtype)
            env_state = GymnasiumState(env_id=jnp.full(shape, env_id, dtype=jnp.int32))
            timestep = Timestep(
                next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                action=jnp.reshape(action_sentinel, shape + action_spec.shape),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.zeros(shape, dtype=jnp.bool),
                truncated=jnp.zeros(shape, dtype=jnp.bool),
                info=jax.tree.map(lambda i: jnp.reshape(i, shape + i.shape[1:]), info),
            )
            return env_state, timestep

        env_state, timestep = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            env_state.env_id,
            vmap_method="broadcast_all",
        )
        return env_state, timestep

    def step(self, key: Key, env_state: GymnasiumState, action: Array) -> tuple[GymnasiumState, Timestep]:
        del key

        def callback(env_id, action):
            global _envs_gymnasium
            shape = env_id.shape
            envs = _envs_gymnasium[np.ravel(env_id)[0]]
            actions = jax.tree.map(
                lambda a: np.asarray(a).reshape(-1, *action.shape[len(shape):]), action
            )
            obs, reward, terminated, truncated, info = envs.step(actions)
            env_state = GymnasiumState(env_id=env_id)
            timestep = Timestep(
                next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                action=jnp.reshape(action, shape + action.shape[len(shape):]),
                reward=jnp.reshape(reward, shape).astype(jnp.float32),
                terminated=jnp.reshape(terminated, shape).astype(jnp.bool),
                truncated=jnp.reshape(truncated, shape).astype(jnp.bool),
                info=jax.tree.map(lambda i: jnp.reshape(i, shape + i.shape[1:]), info),
            )
            return env_state, timestep

        env_state, timestep = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            env_state.env_id,
            action,
            vmap_method="broadcast_all",
        )
        return env_state, timestep


def _gymnasium_to_gxm_space(gymnasium_space: Any) -> Space:
    if isinstance(gymnasium_space, gymnasium.spaces.Discrete):
        return Discrete(int(gymnasium_space.n))
    elif isinstance(gymnasium_space, gymnasium.spaces.Box):
        return Box(jnp.asarray(gymnasium_space.low), jnp.asarray(gymnasium_space.high), gymnasium_space.shape)
    elif isinstance(gymnasium_space, gymnasium.spaces.MultiDiscrete):
        return Tree(tuple(Discrete(int(n)) for n in gymnasium_space.nvec))
    elif isinstance(gymnasium_space, gymnasium.spaces.Dict):
        return Tree({k: _gymnasium_to_gxm_space(v) for k, v in gymnasium_space.spaces.items()})
    elif isinstance(gymnasium_space, gymnasium.spaces.Tuple):
        return Tree(tuple(_gymnasium_to_gxm_space(s) for s in gymnasium_space.spaces))
    else:
        raise NotImplementedError(f"Gymnasium space {gymnasium_space} not supported.")


def make(id: str, **kwargs) -> Environment:
    return GymnasiumAdapter(id, **kwargs)
