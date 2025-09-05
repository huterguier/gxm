from dataclasses import dataclass
from typing import Any

import time
import envpool
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState

envs_envpool = {}


@jax.tree_util.register_dataclass
@dataclass
class EnvpoolState:
    env_id: jax.Array


class EnvpoolEnvironment(Environment):
    id: str
    env_state_shape_dtype: Any
    _num_actions: int

    def __init__(self, id: str):
        self.id = id
        env = envpool.make(self.id, env_type="gym", num_envs=1, seed=0)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, _ = env.step(np.zeros(1, dtype=int))
        env_state = EnvironmentState(
            state=EnvpoolState(env_id=jnp.int32(0)),
            obs=jnp.array(obs),
            reward=jnp.array(reward),
            done=jnp.array(terminated or truncated),
            info={},
        )
        self.env_state_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), env_state
        )
        self._num_actions = int(env.action_space.n)

    def init(self, key: jax.Array) -> EnvironmentState:
        def callback(key):
            global envs_envpool, current_env_id
            shape = key.shape[:-1]
            keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]
            envs = envpool.make(self.id, env_type="gym", num_envs=num_envs)
            obs, _ = envs.reset()
            env_id = len(envs_envpool)
            envs_envpool[env_id] = envs
            env_state = EnvironmentState(
                state=EnvpoolState(env_id=jnp.full((num_envs,), env_id, dtype=jnp.int32)),
                obs=jnp.asarray(obs),
                reward=jnp.zeros((num_envs,), dtype=jnp.float32),
                done=jnp.zeros((num_envs,), dtype=jnp.bool),
                info={},
            )
            env_state = jax.tree.map(
                lambda x: jnp.reshape(x, shape + x.shape[1:]), env_state
            )
            return env_state

        env_state = jax.pure_callback(
            callback,
            self.env_state_shape_dtype,
            key,
            vmap_method="broadcast_all",
        )
        return env_state

    def step(
        self, key: jax.Array, env_state: EnvironmentState, action: jax.Array
    ) -> EnvironmentState:
        def callback(env_id, action):
            global envs_envpool
            shape = env_id.shape
            envs = envs_envpool[np.ravel(env_id)[0]]
            actions = np.reshape(np.asarray(action), (-1,))
            obs, reward, terminated, truncated, _ = envs.step(
                actions
            )
            done = np.logical_or(terminated, truncated)
            env_state = EnvironmentState(
                state=EnvpoolState(env_id=env_id),
                obs=np.reshape(obs, shape + obs.shape[1:]),
                reward=np.reshape(reward, shape),
                done=np.reshape(done, shape),
                info={},
            )
            return env_state

        env_state = jax.pure_callback(
            callback,
            self.env_state_shape_dtype,
            env_state.state.env_id,
            action,
            vmap_method="broadcast_all",
        )

        return env_state

    def reset(self, key: jax.Array, env_state: EnvironmentState) -> EnvironmentState:
        def callback(env_id):
            global envs_envpool
            shape = env_id.shape
            envs = envs_envpool[np.ravel(env_id)[0]]
            obs = envs.reset()
            env_state = EnvironmentState(
                state=EnvpoolState(env_id=env_id),
                obs=np.reshape(obs, shape + obs.shape[1:]),
                reward=np.zeros(shape, dtype=jnp.float32),
                done=jnp.zeros(shape, dtype=jnp.bool_),
                info={},
            )
            env_state = jax.tree.map(
                lambda x: jnp.reshape(x, shape + x.shape[1:]), env_state
            )
            return env_state

        env_state = jax.pure_callback(
            callback,
            self.env_state_shape_dtype,
            env_state.state.env_id,
            vmap_method="broadcast_all",
        )
        return env_state

    @property
    def num_actions(self):
        return self._num_actions


if __name__ == "__main__":
    env = EnvpoolEnvironment("CartPole-v1")
    env_state = env.init(jax.random.key(0))
    env_state = env.step(jax.random.key(0), env_state, jax.numpy.array(0))

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
            action = jax.random.randint(key_action, (1,), 0, env.num_actions)
            env_state = env.step(key, env_state, action)
            return (env_state, key), _

        env_state = env.init(key)
        (env_state, _), _ = jax.lax.scan(step, (env_state, key), length=n_steps)
        return env_state.reward

    key = jax.random.key(0)
    keys = jax.random.split(key, n_envs)

    start = time.time()
    obs_scan = jax.vmap(rollout_scan)(keys)
    print("Time for rollout_scan:", time.time() - start)

    start = time.time()
    obs_for = rollout_for()
    print("Time for rollout_for:", time.time() - start)
