from dataclasses import dataclass
from typing import Any

import envpool
import gym
import gym.spaces
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key

envs_envpool = {}


@jax.tree_util.register_dataclass
@dataclass
class EnvpoolEnvironmentState(EnvironmentState):
    """The state of an Envpool environment."""

    env_id: Array
    """ The ID referencing the Envpool environment instance. """


class EnvpoolEnvironment(Environment[EnvpoolEnvironmentState]):

    envpool_id: str
    """ The Envpool environment ID. """
    return_shape_dtype: Any
    """ The shape and dtype of the returned EnvironmentState and Timestep. """
    kwargs: Any
    """ The kwargs used to create the Envpool environment. """

    def __init__(self, id: str, **kwargs):
        self.id = id
        self.envpool_id = id.split("/", 1)[1]
        env = envpool.make(self.envpool_id, env_type="gym", num_envs=1, **kwargs)
        obs = env.reset()
        obs, reward, done, info = env.step(np.zeros(1, dtype=int))
        env_state = EnvpoolEnvironmentState(env_id=jnp.int32(0))
        timestep = Timestep(
            obs=jnp.array(obs),
            true_obs=jnp.array(obs),
            reward=jnp.array(reward),
            terminated=jnp.array(done),
            truncated=jnp.array(done),
            info=jax.tree.map(jnp.array, info),
        )
        self.return_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), (env_state, timestep)
        )
        self.action_space = self.envpool_to_gxm_space(env.action_space)
        self.observation_space = self.envpool_to_gxm_space(env.observation_space)
        self.kwargs = kwargs

    def init(self, key: Key) -> tuple[EnvpoolEnvironmentState, Timestep]:
        def callback(key):
            global envs_envpool, current_env_id
            shape = key.shape[:-1]
            keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]
            envs = envpool.make(
                self.envpool_id, env_type="gym", num_envs=num_envs, **self.kwargs
            )
            obs = envs.reset()
            env_id = len(envs_envpool)
            envs_envpool[env_id] = envs
            env_state = EnvpoolEnvironmentState(
                env_id=jnp.full(shape, env_id, dtype=jnp.int32)
            )

            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.zeros(shape, dtype=jnp.bool),
                truncated=jnp.zeros(shape, dtype=jnp.bool),
                info=jax.tree.map(
                    lambda ds: jnp.zeros(shape + ds.shape[:1], dtype=ds.dtype),
                    self.return_shape_dtype[1].info,
                ),
            )
            return env_state, timestep

        env_state, timestep = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            jax.random.key_data(key),
            vmap_method="broadcast_all",
        )
        return env_state, timestep

    def step(
        self, key: Key, env_state: EnvpoolEnvironmentState, action: Array
    ) -> tuple[EnvpoolEnvironmentState, Timestep]:
        del key

        def callback(env_id, action):
            global envs_envpool
            shape = env_id.shape
            envs = envs_envpool[np.ravel(env_id)[0]]
            actions = np.reshape(np.asarray(action), (-1,))
            obs, reward, done, info = envs.step(actions)
            env_state = EnvpoolEnvironmentState(env_id=env_id)
            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.reshape(reward, shape),
                terminated=jnp.reshape(done, shape),
                truncated=jnp.full_like(jnp.reshape(done, shape), False),
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

    def reset(
        self, key: Key, env_state: EnvpoolEnvironmentState
    ) -> tuple[EnvpoolEnvironmentState, Timestep]:
        del key

        def callback(env_id):
            global envs_envpool
            shape = env_id.shape
            envs = envs_envpool[np.ravel(env_id)[0]]
            obs = envs.reset()
            env_state = EnvpoolEnvironmentState(
                env_id=jnp.full(shape, env_id, dtype=jnp.int32)
            )
            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.zeros(shape, dtype=jnp.bool),
                truncated=jnp.zeros(shape, dtype=jnp.bool),
                info=jax.tree.map(
                    lambda ds: jnp.zeros(shape + ds.shape[:1], dtype=ds.dtype),
                    self.return_shape_dtype[1].info,
                ),
            )
            return env_state, timestep

        env_state, timestep = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            env_state.env_id,
            vmap_method="broadcast_all",
        )
        return env_state, timestep

    @classmethod
    def envpool_to_gxm_space(cls, envpool_space: Any) -> Space:
        if isinstance(envpool_space, gym.spaces.Discrete):
            return Discrete(int(envpool_space.n))
        elif isinstance(envpool_space, gym.spaces.Box):
            return Box(
                jnp.asarray(envpool_space.low),
                jnp.asarray(envpool_space.high),
                envpool_space.shape,
            )
        elif isinstance(envpool_space, gym.spaces.Dict):
            return Tree(
                {
                    k: cls.envpool_to_gxm_space(v)
                    for k, v in envpool_space.spaces.items()
                }
            )
        elif isinstance(envpool_space, gym.spaces.Tuple):
            return Tree(
                tuple(cls.envpool_to_gxm_space(s) for s in envpool_space.spaces)
            )
        else:
            raise NotImplementedError(f"Envpool space {envpool_space} not supported.")
