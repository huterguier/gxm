import time
from dataclasses import dataclass
from typing import Any

import envpool
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree

envs_envpool = {}


def envpool_to_gxm_space(envpool_space: Any) -> Space:
    if isinstance(envpool_space, gymnasium.spaces.Discrete):
        return Discrete(int(envpool_space.n))
    elif isinstance(envpool_space, gymnasium.spaces.Box):
        return Box(
            jnp.asarray(envpool_space.low),
            jnp.asarray(envpool_space.high),
            envpool_space.shape,
        )
    elif isinstance(envpool_space, gymnasium.spaces.Dict):
        return Tree(
            {k: envpool_to_gxm_space(v) for k, v in envpool_space.spaces.items()}
        )
    elif isinstance(envpool_space, gymnasium.spaces.Tuple):
        return Tree(tuple(envpool_to_gxm_space(s) for s in envpool_space.spaces))
    else:
        raise NotImplementedError(f"Envpool space {envpool_space} not supported.")


@jax.tree_util.register_dataclass
@dataclass
class EnvpoolState:
    env_id: jax.Array


class EnvpoolEnvironment(Environment):
    id: str
    return_shape_dtype: Any
    kwargs: Any

    def __init__(self, id: str, **kwargs):
        self.id = id
        env = envpool.make(self.id, env_type="gymnasium", num_envs=1, **kwargs)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, _ = env.step(np.zeros(1, dtype=int))
        env_state = EnvpoolState(env_id=jnp.int32(0))
        timestep = Timestep(
            obs=jnp.array(obs),
            true_obs=jnp.array(obs),
            reward=jnp.array(reward),
            terminated=jnp.array(terminated),
            truncated=jnp.array(truncated),
            info={},
        )
        self.return_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), (env_state, timestep)
        )
        self.action_space = envpool_to_gxm_space(env.action_space)
        self.kwargs = kwargs

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        def callback(key):
            global envs_envpool, current_env_id
            shape = key.shape[:-1]
            keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]
            envs = envpool.make(
                self.id, env_type="gymnasium", num_envs=num_envs, **self.kwargs
            )
            obs, _ = envs.reset()
            env_id = len(envs_envpool)
            envs_envpool[env_id] = envs
            env_state = (EnvpoolState(env_id=jnp.full(shape, env_id, dtype=jnp.int32)),)
            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.zeros(shape, dtype=jnp.bool),
                truncated=jnp.zeros(shape, dtype=jnp.bool),
                info={},
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
        self, key: jax.Array, env_state: EnvironmentState, action: jax.Array
    ) -> tuple[EnvironmentState, Timestep]:
        def callback(env_id, action):
            global envs_envpool
            shape = env_id.shape
            envs = envs_envpool[np.ravel(env_id)[0]]
            actions = np.reshape(np.asarray(action), (-1,))
            obs, reward, terminated, truncated, _ = envs.step(actions)
            env_state = (EnvpoolState(env_id=env_id),)
            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.reshape(reward, shape),
                terminated=jnp.reshape(terminated, shape),
                truncated=jnp.reshape(truncated, shape),
                info={},
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

    def reset(self, key: jax.Array, env_state: EnvironmentState) -> EnvironmentState:
        def callback(env_id):
            global envs_envpool
            shape = env_id.shape
            envs = envs_envpool[np.ravel(env_id)[0]]
            obs, _ = envs.reset()
            env_state = (EnvpoolState(env_id=jnp.full(shape, env_id, dtype=jnp.int32)),)
            timestep = Timestep(
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.zeros(shape, dtype=jnp.bool),
                truncated=jnp.zeros(shape, dtype=jnp.bool),
                info={},
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
        if isinstance(envpool_space, gymnasium.spaces.Discrete):
            return Discrete(int(envpool_space.n))
        elif isinstance(envpool_space, gymnasium.spaces.Box):
            return Box(
                jnp.asarray(envpool_space.low),
                jnp.asarray(envpool_space.high),
                envpool_space.shape,
            )
        elif isinstance(envpool_space, gymnasium.spaces.Dict):
            return Tree(
                {
                    k: cls.envpool_to_gxm_space(v)
                    for k, v in envpool_space.spaces.items()
                }
            )
        elif isinstance(envpool_space, gymnasium.spaces.Tuple):
            return Tree(
                tuple(cls.envpool_to_gxm_space(s) for s in envpool_space.spaces)
            )
        else:
            raise NotImplementedError(f"Envpool space {envpool_space} not supported.")


if __name__ == "__main__":
    env = EnvpoolEnvironment("Breakout-v5")
    env_state, timestep = env.init(jax.random.key(0))
    env_state, timestep = env.step(jax.random.key(0), env_state, jax.numpy.array(0))

    n_envs = 8
    n_steps = 1000

    def rollout_for():
        env = envpool.make("Breakout-v5", env_type="gymnasium", num_envs=n_envs)
        reward = None
        for _ in range(n_steps):
            action = np.random.randint(0, env.action_space.n, size=(n_envs,))
            obs, reward, terminated, truncated, _ = env.step(action)
        return reward

    @jax.jit
    def rollout_scan(key):
        def step(carry, _):
            env_state, key = carry
            key, key_action, key_step = jax.random.split(key, 3)
            action = env.action_space.sample(key)
            env_state, timestep = env.step(key, env_state, action)
            return (env_state, key), _

        env_state, timestep = env.init(key)
        (env_state, _), _ = jax.lax.scan(step, (env_state, key), length=n_steps)
        return timestep.reward

    key = jax.random.key(0)
    keys = jax.random.split(key, n_envs)

    start = time.time()
    obs_scan = jax.vmap(rollout_scan)(keys)
    print("Time for rollout_scan:", time.time() - start)

    start = time.time()
    obs_for = rollout_for()
    print("Time for rollout_for:", time.time() - start)
