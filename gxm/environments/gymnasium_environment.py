from dataclasses import dataclass
from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key

envs_gymnasium = {}


@jax.tree_util.register_dataclass
@dataclass
class GymnasiumState(EnvironmentState):
    env_id: Array


class GymnasiumEnvironment(Environment[GymnasiumState]):

    gymnasium_id: str
    """ The Gymnasium environment ID. """
    return_shape_dtype: Any
    """ The shape and dtype of the returned EnvironmentState and Timestep. """
    kwargs: Any
    """ The kwargs used to create the Gymnasium environment. """

    def __init__(self, id: str, **kwargs):
        self.id = id
        self.gymnasium_id = id.split("/", 1)[1]
        env = gymnasium.make_vec(self.gymnasium_id, num_envs=1, **kwargs)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env_state = GymnasiumState(env_id=jnp.int32(0))
        timestep = Timestep(
            obs=jnp.array(obs),
            true_obs=jnp.array(obs),
            reward=jnp.array(reward, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool),
            truncated=jnp.array(truncated, dtype=jnp.bool),
            info=jax.tree.map(jnp.array, info),
        )
        self.return_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), (env_state, timestep)
        )
        self.action_space = self.gymnasium_to_gxm_space(env.single_action_space)
        self.observation_space = self.gymnasium_to_gxm_space(
            env.single_observation_space
        )
        self.kwargs = kwargs

    def init(self, key: Key) -> tuple[GymnasiumState, Timestep]:
        def callback(key):
            global envs_gymnasium, current_env_id
            shape = key.shape[:-1]
            keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]
            envs = gymnasium.make_vec(
                self.gymnasium_id, num_envs=num_envs, **self.kwargs
            )
            obs, info = envs.reset(seed=0)
            env_id = len(envs_gymnasium)
            envs_gymnasium[env_id] = envs
            env_state = GymnasiumState(env_id=jnp.full(shape, env_id, dtype=jnp.int32))
            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
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

    def reset(
        self, key: Key, env_state: GymnasiumState
    ) -> tuple[GymnasiumState, Timestep]:
        del key

        def callback(env_id):
            global envs_gymnasium
            shape = env_id.shape
            envs = envs_gymnasium[np.ravel(env_id)[0]]
            obs, info = envs.reset()
            env_state = GymnasiumState(env_id=jnp.full(shape, env_id, dtype=jnp.int32))
            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
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

    def step(
        self, key: Key, env_state: GymnasiumState, action: Array
    ) -> tuple[GymnasiumState, Timestep]:
        del key

        def callback(env_id, action):
            global envs_gymnasium
            shape = env_id.shape
            envs = envs_gymnasium[np.ravel(env_id)[0]]
            actions = np.reshape(np.asarray(action), (-1,))
            obs, reward, terminated, truncated, info = envs.step(actions)
            env_state = GymnasiumState(env_id=env_id)
            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
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

    @classmethod
    def gymnasium_to_gxm_space(cls, gymnasium_space: Any) -> Space:
        """Convert a Gymnasium space to a Gxm space."""
        if isinstance(gymnasium_space, gymnasium.spaces.Discrete):
            return Discrete(int(gymnasium_space.n))
        elif isinstance(gymnasium_space, gymnasium.spaces.Box):
            return Box(
                jnp.asarray(gymnasium_space.low),
                jnp.asarray(gymnasium_space.high),
                gymnasium_space.shape,
            )
        elif isinstance(gymnasium_space, gymnasium.spaces.MultiDiscrete):
            return Tree(tuple(Discrete(int(n)) for n in gymnasium_space.nvec))
        elif isinstance(gymnasium_space, gymnasium.spaces.Dict):
            return Tree(
                {
                    k: cls.gymnasium_to_gxm_space(v)
                    for k, v in gymnasium_space.spaces.items()
                }
            )
        elif isinstance(gymnasium_space, gymnasium.spaces.Tuple):
            return Tree(
                tuple(cls.gymnasium_to_gxm_space(s) for s in gymnasium_space.spaces)
            )
        else:
            raise NotImplementedError(
                f"Gymnasium space {gymnasium_space} not supported."
            )
