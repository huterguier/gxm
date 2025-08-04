from dataclasses import dataclass
from typing import Any

import gymnasium
import jax
import jax.numpy as jnp

from gxm.core import Env, EnvState, State

envs = {}


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
        n_actions = env.action_space.n

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
            print(state)
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
            env_id = 0
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

    def num_actions():
        return self.n_actions


env = GymnasiumEnv("CartPole-v1")
env_state = env.reset(jax.random.key(0))
env_state = env.step(jax.random.key(0), env_state, jax.numpy.array(0))
