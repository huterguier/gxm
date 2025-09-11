import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState

envs_gymnasium = {}


@jax.tree_util.register_dataclass
@dataclass
class GymnasiumState:
    env_id: jax.Array


class GymnasiumEnvironment(Environment):
    id: str
    env_state_shape_dtype: Any
    _num_actions: int
    kwargs: Any

    def __init__(self, id: str, **kwargs):
        self.id = id
        env = gym.make_vec(self.id, num_envs=1, **kwargs)
        obs, _ = env.reset()
        zero_act = np.zeros(
            (1,),
            dtype=(
                env.single_action_space.dtype
                if hasattr(env, "single_action_space")
                else int
            ),
        )
        obs, reward, terminated, truncated, _ = env.step(zero_act)

        env_state = EnvironmentState(
            state=GymnasiumState(env_id=jnp.int32(0)),
            obs=jnp.array(obs),
            true_obs=jnp.array(obs),
            reward=jnp.array(reward),
            terminated=jnp.array(terminated),
            truncated=jnp.array(truncated),
            info={},
        )
        self.env_state_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), env_state
        )
        base_space = getattr(env, "single_action_space", env.action_space)
        assert hasattr(
            base_space, "n"
        ), "GymnasiumEnvironment expects a discrete action space."
        self._num_actions = int(base_space.n)
        self.kwargs = kwargs

    def init(self, key: jax.Array) -> EnvironmentState:
        def callback(key):
            global envs_gymnasium
            shape = key.shape[:-1]
            keys_flat = np.reshape(np.asarray(key), (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]

            envs = gym.make_vec(self.id, num_envs=num_envs, **self.kwargs)
            obs, _ = envs.reset()

            env_id = len(envs_gymnasium)
            envs_gymnasium[env_id] = envs

            env_state = EnvironmentState(
                state=GymnasiumState(env_id=jnp.full(shape, env_id, dtype=jnp.int32)),
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.zeros(shape, dtype=jnp.bool_),
                truncated=jnp.zeros(shape, dtype=jnp.bool_),
                info={},
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
            global envs_gymnasium
            shape = env_id.shape
            envs = envs_gymnasium[np.ravel(env_id)[0]]

            actions = np.reshape(np.asarray(action), (-1,))
            obs, reward, terminated, truncated, _ = envs.step(actions)

            new_state = EnvironmentState(
                state=GymnasiumState(env_id=env_id),
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.reshape(reward, shape),
                terminated=jnp.reshape(terminated, shape),
                truncated=jnp.reshape(truncated, shape),
                info={},
            )
            return new_state

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
            global envs_gymnasium
            shape = env_id.shape
            envs = envs_gymnasium[np.ravel(env_id)[0]]
            obs, _ = envs.reset()

            new_state = EnvironmentState(
                state=GymnasiumState(env_id=jnp.full(shape, env_id, dtype=jnp.int32)),
                obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.zeros(shape, dtype=jnp.bool_),
                truncated=jnp.zeros(shape, dtype=jnp.bool_),
                info={},
            )
            return new_state

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
    env = GymnasiumEnvironment("CartPole-v1")
    env_state = env.init(jax.random.key(0))
    env_state = env.step(jax.random.key(0), env_state, jnp.array(0))

    n_envs = 8
    n_steps = 1_000

    def rollout_for():
        env = gym.make_vec("CartPole-v1", num_envs=n_envs)
        obs, _ = env.reset()
        reward = None
        for _ in range(n_steps):
            action = np.random.randint(0, env.single_action_space.n, size=(n_envs,))
            obs, reward, terminated, truncated, _ = env.step(action)
        return reward

    @jax.jit
    def rollout_scan(key):
        def step(carry, _):
            env_state, key = carry
            key, key_action = jax.random.split(key, 2)
            action = jax.random.randint(key_action, (1,), 0, env.num_actions)
            env_state = env.step(key, env_state, action)
            return (env_state, key), None

        env_state = env.init(key)
        (env_state, _), _ = jax.lax.scan(step, (env_state, key), length=n_steps)
        return env_state.reward

    key = jax.random.key(0)
    keys = jax.random.split(key, n_envs)

    start = time.time()
    rew_scan = jax.vmap(rollout_scan)(keys)
    print("Time for rollout_scan:", time.time() - start)

    start = time.time()
    rew_for = rollout_for()
    print("Time for rollout_for:", time.time() - start)
