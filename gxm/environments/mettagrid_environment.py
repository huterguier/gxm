from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space
from gxm.typing import Array, Key

envs_mettagrid: dict[int, Any] = {}


@jax.tree_util.register_dataclass
@dataclass
class MettagridState(EnvironmentState):
    env_id: Array


class MettagridEnvironment(Environment[MettagridState]):
    mettagrid_id: str
    return_shape_dtype: Any
    num_agents: int
    kwargs: dict[str, Any]

    def __init__(self, id: str, num_agents: int = 24, **kwargs):
        from mettagrid.builder.envs import make_arena
        from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
        from mettagrid.simulator import Simulator

        self.id = id
        self.mettagrid_id = id.split("/", 1)[1]
        self.num_agents = num_agents
        self.kwargs = kwargs

        simulator = Simulator()
        cfg = make_arena(num_agents=num_agents, **kwargs)
        env = MettaGridPufferEnv(simulator=simulator, cfg=cfg)

        obs, _ = env.reset(seed=0)
        action = np.zeros(env.num_agents, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()

        self.action_space = self.mettagrid_to_gxm_space(env.single_action_space)
        self.observation_space = self.mettagrid_to_gxm_space(
            env.single_observation_space
        )

        env_state = MettagridState(env_id=jnp.array([0], dtype=jnp.int32))
        timestep = Timestep(
            obs=jnp.array(obs)[None],
            true_obs=jnp.array(obs)[None],
            reward=jnp.array(reward, dtype=jnp.float32)[None],
            terminated=jnp.array(terminated, dtype=jnp.bool)[None],
            truncated=jnp.array(truncated, dtype=jnp.bool)[None],
            info={},
        )
        self.return_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), (env_state, timestep)
        )

    def init(self, key: Key) -> tuple[MettagridState, Timestep]:
        from mettagrid.builder.envs import make_arena
        from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
        from mettagrid.simulator import Simulator

        def callback(key):
            global envs_mettagrid
            shape = key.shape[:-1]
            keys_flat = np.reshape(np.asarray(key), (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]

            all_obs = []
            env_ids = []
            for i in range(num_envs):
                seed = int(keys_flat[i, 0])
                simulator = Simulator()
                cfg = make_arena(num_agents=self.num_agents, **self.kwargs)
                env = MettaGridPufferEnv(simulator=simulator, cfg=cfg, seed=seed)
                obs, _ = env.reset(seed=seed)

                env_id = len(envs_mettagrid)
                envs_mettagrid[env_id] = env
                env_ids.append(env_id)
                all_obs.append(obs)

            obs_stacked = np.stack(all_obs, axis=0)

            env_state = MettagridState(
                env_id=jnp.reshape(jnp.array(env_ids, dtype=jnp.int32), shape)
            )
            timestep = Timestep(
                obs=jnp.reshape(jnp.array(obs_stacked), shape + obs_stacked.shape[1:]),
                true_obs=jnp.reshape(
                    jnp.array(obs_stacked), shape + obs_stacked.shape[1:]
                ),
                reward=jnp.zeros(shape + (self.num_agents,), dtype=jnp.float32),
                terminated=jnp.zeros(shape + (self.num_agents,), dtype=jnp.bool),
                truncated=jnp.zeros(shape + (self.num_agents,), dtype=jnp.bool),
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

    def reset(
        self, key: Key, env_state: MettagridState
    ) -> tuple[MettagridState, Timestep]:
        del key

        def callback(env_id):
            global envs_mettagrid
            shape = env_id.shape
            env_ids_flat = np.ravel(np.asarray(env_id))
            num_envs = env_ids_flat.shape[0]

            all_obs = []
            for i in range(num_envs):
                env = envs_mettagrid[int(env_ids_flat[i])]
                obs, _ = env.reset()
                all_obs.append(obs)

            obs_stacked = np.stack(all_obs, axis=0)

            env_state = MettagridState(env_id=jnp.array(env_id, dtype=jnp.int32))
            timestep = Timestep(
                obs=jnp.reshape(jnp.array(obs_stacked), shape + obs_stacked.shape[1:]),
                true_obs=jnp.reshape(
                    jnp.array(obs_stacked), shape + obs_stacked.shape[1:]
                ),
                reward=jnp.zeros(shape + (self.num_agents,), dtype=jnp.float32),
                terminated=jnp.zeros(shape + (self.num_agents,), dtype=jnp.bool),
                truncated=jnp.zeros(shape + (self.num_agents,), dtype=jnp.bool),
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

    def step(
        self, key: Key, env_state: MettagridState, action: Array
    ) -> tuple[MettagridState, Timestep]:
        del key

        def callback(env_id, action):
            global envs_mettagrid
            shape = env_id.shape
            env_ids_flat = np.ravel(np.asarray(env_id))
            num_envs = env_ids_flat.shape[0]

            actions_np = np.asarray(action, dtype=np.int32)
            if actions_np.ndim == 1:
                actions_np = actions_np[None, :]
            else:
                actions_np = np.reshape(actions_np, (num_envs, -1))

            all_obs = []
            all_rewards = []
            all_terminated = []
            all_truncated = []

            for i in range(num_envs):
                env = envs_mettagrid[int(env_ids_flat[i])]
                obs, reward, terminated, truncated, info = env.step(actions_np[i])
                all_obs.append(obs)
                all_rewards.append(reward)
                all_terminated.append(terminated)
                all_truncated.append(truncated)

            obs_stacked = np.stack(all_obs, axis=0)
            reward_stacked = np.stack(all_rewards, axis=0)
            terminated_stacked = np.stack(all_terminated, axis=0)
            truncated_stacked = np.stack(all_truncated, axis=0)

            env_state = MettagridState(env_id=env_id)
            timestep = Timestep(
                obs=jnp.reshape(jnp.array(obs_stacked), shape + obs_stacked.shape[1:]),
                true_obs=jnp.reshape(
                    jnp.array(obs_stacked), shape + obs_stacked.shape[1:]
                ),
                reward=jnp.reshape(
                    jnp.array(reward_stacked, dtype=jnp.float32),
                    shape + reward_stacked.shape[1:],
                ),
                terminated=jnp.reshape(
                    jnp.array(terminated_stacked, dtype=jnp.bool),
                    shape + terminated_stacked.shape[1:],
                ),
                truncated=jnp.reshape(
                    jnp.array(truncated_stacked, dtype=jnp.bool),
                    shape + truncated_stacked.shape[1:],
                ),
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

    @classmethod
    def mettagrid_to_gxm_space(cls, space: Any) -> Space:
        from gymnasium.spaces import Box as GymBox
        from gymnasium.spaces import Discrete as GymDiscrete

        if isinstance(space, GymDiscrete):
            return Discrete(int(space.n))
        elif isinstance(space, GymBox):
            return Box(
                jnp.asarray(space.low),
                jnp.asarray(space.high),
                space.shape,
            )
        else:
            raise NotImplementedError(f"Space type {type(space)} not supported.")
