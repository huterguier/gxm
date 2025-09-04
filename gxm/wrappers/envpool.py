from dataclasses import dataclass
from typing import Any

import envpool
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState

envs_envpool = {}
current_env_id = 0


@dataclass
class EnvpoolState:
    time: jax.Array
    env_id: jax.Array


class GymnasiumEnvironment(Environment):
    id: str
    env_state_shape_dtype: Any
    _num_actions: int

    def __init__(self, id: str, **kwargs):
        self.id = id
        env = envpool.make(self.id, **kwargs)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env_state = EnvironmentState(
            state=EnvpoolState(time=jnp.int32(0), env_id=jnp.int32(0)),
            obs=jnp.array(obs),
            reward=jnp.array(reward),
            done=jnp.array(terminated or truncated),
            info={},
        )
        self.env_state_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), env_state
        )
        self.n_actions = env.action_space.n

    def init(self, key: jax.Array) -> EnvironmentState:
        keys = key
        shape_keys = key.shape[:-1]
        keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
        num_envs = keys_flat.shape[0]
        envs = envpool.make(self.id, env_type="gymnasium", num_envs=num_envs)
        obs, _ = envs.reset()
        env_id = len(envs_envpool)
        env_state = EnvironmentState(
            state=EnvpoolState(
                time=jnp.zeros((num_envs,), dtype=jnp.int32),
                env_id=jnp.full((num_envs,), env_id, dtype=jnp.int32),
            ),
            obs=jnp.array(obs),
            reward=jnp.zeros((num_envs,), dtype=jnp.float32),
            done=jnp.zeros((num_envs,), dtype=jnp.bool),
            info={},
        )
        return env_state

    def step(
        self, key: jax.Array, state: EnvState | EnvpoolState, action: jax.Array
    ) -> EnvState:

        def callback(key: jax.Array, state: GymnasiumState, action: jax.Array):
            def flatten(x, batch_shape):
                return jnp.reshape(x, (-1,) + x.shape[len(batch_shape) :])

            def unflatten(x_flat, batch_shape):
                return jnp.reshape(x_flat, batch_shape + x_flat.shape[1:])

            batch_shape = key.shape[:-1]
            keys = flatten(key, batch_shape)
            state = jax.tree.map(lambda x: flatten(x, batch_shape), state)
            action = flatten(action, batch_shape)
            env_ids = state.env_id

            obss, rewardss, dones, infoss = [], [], [], []
            for action, env_id in zip(action, env_ids):
                env = envs.get(str(env_id), None)
                if env is None:
                    raise ValueError(f"Environment with id {env_id} not found.")
                obs, reward, terminated, truncated, info = env.step(np.array(action))
                obss.append(obs)
                rewardss.append(reward)
                dones.append(terminated or truncated)
                infoss.append(info)

            env_state = EnvState(
                state=GymnasiumState(
                    time=state.time + 1,
                    env_id=env_ids,
                ),
                obs=jnp.stack(obss),
                reward=jnp.array(rewardss),
                done=jnp.array(dones),
                info=jax.tree.map(lambda x: jnp.stack(x), infoss),
            )

            return jax.tree.map(lambda x: unflatten(x, batch_shape), env_state)

            env = envs.get(str(state.env_id), None)

            obs, reward, terminated, truncated, info = env.step(np.array(action))
            obs = jax.numpy.array(obs)
            env_state = EnvState(
                state=state,
                obs=jnp.array(obs),
                reward=jnp.array(reward),
                done=jnp.array(terminated or truncated),
                info=info,
            )

            return env_state

        env_state = jax.pure_callback(
            callback,
            self.env_state_shape_dtype,
            key,
            state,
            action,
            vmap_method="broadcast_all",
        )

        return env_state

    def reset(self, key: jax.Array) -> EnvState:
        def callback(key: jax.Array):
            def flatten(x, batch_shape):
                return jnp.reshape(x, (-1,) + x.shape[len(batch_shape) :])

            def unflatten(x_flat, batch_shape):
                return jnp.reshape(x_flat, batch_shape + x_flat.shape[1:])

            batch_shape = key.shape[:-1]
            keys = flatten(key, batch_shape)

            global current_env_id
            env_ids = jnp.arange(
                current_env_id, current_env_id + len(keys), dtype=jnp.int32
            )
            current_env_id += len(keys)
            obss, infoss = [], []
            for env_id in env_ids:
                env = gymnasium.make(self.id)
                env = gymnasium.wrappers.Autoreset(env)
                envs[str(env_id)] = env
                obs, info = env.reset()
                obss.append(obs)
                infoss.append(info)

            state = GymnasiumState(
                time=jnp.zeros((len(keys),), dtype=jnp.int32),
                env_id=env_ids,
            )
            env_state = EnvState(
                state=state,
                obs=jnp.stack(obss),
                reward=jnp.zeros((len(keys),), dtype=jnp.float32),
                done=jnp.zeros((len(keys),), dtype=jnp.bool),
                info=jax.tree.map(lambda x: jnp.stack(x), infoss),
            )

            return jax.tree.map(lambda x: unflatten(x, batch_shape), env_state)

        batch_shape = key.shape[:-1]
        env_state = jax.pure_callback(
            callback,
            self.env_state_shape_dtype,
            key,
            vmap_method="broadcast_all",
        )
        return env_state

    @property
    def num_actions(self):
        return self.n_actions


if __name__ == "__main__":
    env = GymnasiumEnv("CartPole-v1")
    env_state = env.reset(jax.random.key(0))
    env_state = env.step(jax.random.key(0), env_state, jax.numpy.array(0))

    n_steps = 10000
    key = jax.random.key(0)

    @jax.jit
    def f(x, key):
        key, subkey = jax.random.split(key)
        x += jax.random.normal(subkey, x.shape)
        return x, key

    import numpy as np

    @jax.jit
    def f(x, key):
        key, subkey = jax.random.split(key)
        x += jax.random.normal(subkey, x.shape)
        return x, key

    def rollout_for(key):
        env = gymnasium.make("CartPole-v1")
        env = gymnasium.wrappers.Autoreset(env)
        obs, info = env.reset()
        x = jnp.array(0.0)
        for i in range(n_steps):
            obs, reward, termination, trunction, info = env.step(
                np.array(jax.random.randint(key, (), 0, env.action_space.n))
            )
            x, key = f(x, key)
        return obs

    @jax.jit
    def rollout_scan(key):
        def step(env_state, key):
            action = jax.random.randint(key, (), 0, env.num_actions)
            return env.step(key, env_state, action), None

        env_state = env.reset(key)
        keys = jax.random.split(key, n_steps)
        env_state, _ = jax.lax.scan(step, env_state, keys)
        return env_state.obs

    import time

    start = time.time()
    obs_for = rollout_for(key)
    print("Time for rollout_for:", time.time() - start)

    start = time.time()
    obs_scan = rollout_scan(key)
    print("Time for rollout_scan:", time.time() - start)

    start = time.time()
    obs_scan = rollout_scan(key)
    print("Time for rollout_scan again:", time.time() - start)
