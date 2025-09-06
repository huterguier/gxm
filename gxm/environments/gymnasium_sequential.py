from dataclasses import dataclass
from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Env, EnvState, State

envs = {}
current_env_id = 0


@jax.tree_util.register_dataclass
@dataclass
class GymnasiumSequentialState(State):
    env_id: jax.Array


class GymnasiumSequentialEnv(Env):
    id: str
    env_state_shape_dtype: Any
    n_actions: int

    def __init__(self, id: str, **kwargs):
        self.id = id
        env = gymnasium.make(self.id, **kwargs)
        obs, _ = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env_state = EnvState(
            state=GymnasiumSequentialState(time=jnp.int32(0), env_id=jnp.int32(0)),
            obs=jnp.asarray(obs),
            reward=jnp.asarray(reward),
            done=jnp.asarray(terminated or truncated),
            info={},
        )
        self.env_state_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), env_state
        )
        self.n_actions = env.action_space.n

    def step(
        self,
        key: jax.Array,
        state: EnvState | GymnasiumSequentialState,
        action: jax.Array,
    ) -> EnvState:
        if isinstance(state, EnvState):
            env_state = state
            state = env_state.state

        def callback(
            key: jax.Array, state: GymnasiumSequentialState, action: jax.Array
        ):
            env = envs[int(state.env_id)]
            obs, reward, terminated, truncated, info = env.step(np.asarray(action))
            env_state = EnvState(
                state=state,
                obs=jnp.asarray(obs),
                reward=jnp.asarray(reward),
                done=jnp.asarray(terminated or truncated),
                info={},
            )
            return env_state

        env_state = jax.pure_callback(
            callback,
            self.env_state_shape_dtype,
            key,
            state,
            action,
            vmap_method="sequential",
        )

        return env_state

    def reset(self, key: jax.Array) -> EnvState:
        def callback(key: jax.Array):
            env_id = len(envs)
            state = GymnasiumSequentialState(
                time=0,
                env_id=env_id,
            )
            envs[env_id] = gymnasium.make(self.id)
            obs, _ = envs[env_id].reset()
            env_state = EnvState(
                state=state,
                obs=jnp.asarray(obs),
                reward=jnp.asarray(0.0, dtype=jnp.float32),
                done=jnp.asarray(False, dtype=jnp.bool_),
                info={},
            )

            return env_state

        batch_shape = key.shape[:-1]
        env_state = jax.pure_callback(
            callback,
            self.env_state_shape_dtype,
            key,
            vmap_method="sequential",
        )
        return env_state

    @property
    def num_actions(self):
        return self.n_actions


if __name__ == "__main__":
    env = GymnasiumSequentialEnv("CartPole-v1")
    env_state = env.reset(jax.random.key(0))
    env_state = env.step(jax.random.key(0), env_state, jax.numpy.array(0))

    n_steps = 100000
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
