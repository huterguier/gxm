from dataclasses import dataclass
from typing import Any

import gymnasium
import jax
import jax.numpy as jnp

from gxm.core import Env, EnvState, State

envs = {}
current_env_id = 0


@jax.tree_util.register_dataclass
@dataclass
class GymnasiumState(State):
    id: jax.Array


class GymnasiumEnv(Env):
    id: str
    env_state_shape_dtype: Any
    n_actions: int

    def __init__(self, id: str, **kwargs):
        self.id = id
        env = gymnasium.make(self.id, **kwargs)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env_state = EnvState(
            state=GymnasiumState(time=jnp.int32(0), id=jnp.int32(0)),
            obs=jnp.array(obs),
            reward=jnp.array(reward),
            done=jnp.array(terminated or truncated),
            info=info,
        )
        self.env_state_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), env_state
        )
        self.n_actions = env.action_space.n

    def step(
        self, key: jax.Array, state: EnvState | GymnasiumState, action: jax.Array
    ) -> EnvState:
        if isinstance(state, EnvState):
            env_state = state
            state = env_state.state

        def callback(key: jax.Array, state: GymnasiumState, action: jax.Array):
            env = envs.get(str(state.id), None)
            import numpy as np

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
            env = gymnasium.make("CartPole-v1")
            env = gymnasium.wrappers.Autoreset(env)
            global current_env_id
            env_id = current_env_id
            current_env_id += 1

            envs[str(env_id)] = env
            obs, info = env.reset()
            obs = jnp.array(obs)
            env_state = EnvState(
                state=GymnasiumState(time=jnp.int32(0), id=jnp.int32(env_id)),
                obs=jnp.array(obs),
                reward=jnp.array(0.0),
                done=jnp.array(False),
                info=info,
            )
            return env_state

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
