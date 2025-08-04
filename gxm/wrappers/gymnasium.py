from dataclasses import dataclass
from typing import Any

import gymnasium
import jax

from gxm.core import Env, EnvState, State

envs = {}


@jax.tree_util.register_dataclass
@dataclass
class GymnasiumState(State):
    id: jax.Array


class GymnasiumEnv(Env):
    obs_shape_dtype: Any
    n_actions: int

    def __init__(self, id: str, **kwargs):
        env = gymnasium.make(id, **kwargs)
        obs, _ = env.reset()
        obs = jax.numpy.array(obs)
        n_actionss = env.action_space.n
        self.obs_shape_dtype = jax.ShapeDtypeStruct(obs.shape, obs.dtype)

    def step(
        self, key: jax.Array, state: EnvState | GymnasiumState, action: jax.Array
    ) -> EnvState:

        def callback(key: jax.Array, state: GymnasiumState, action: jax.Array):
            env = gymnasium.make("CartPole-v1")
            obs, reward, terminated, truncated, info = env.step(action)
            obs = jax.numpy.array(obs)
            return EnvState(
                state=state,
                obs=obs,
                reward=reward,
                done=done,
                info=info,
            )

        env_state = jax.pure_callback(
            callback,
            self.obs_shape_dtype,
            key=key,
            state=state,
            action=action,
            name="gymnasium_step",
            vmap_method=
        )


        pass

    def reset(key: jax.Array) -> EnvState:
        def callback():

        pass

    def num_actions():
        return 2


env = GymnasiumEnv("CartPole-v1")
