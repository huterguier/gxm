from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import pgx
from pgx.experimental import auto_reset

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete
from gxm.typing import Array, Key


@jax.tree_util.register_dataclass
@dataclass
class PgxState(EnvironmentState):
    pgx_state: pgx.State


class PgxAdapter(Environment[PgxState]):
    pgx_id: str
    env: pgx.Env

    def __init__(self, id: str, **kwargs):
        self.pgx_id = id
        self.id = f"Pgx/{id}"
        self.env = pgx.make(cast(pgx.EnvId, id), **kwargs)
        self.action_space = Discrete(self.env.num_actions)
        self.observation_space = Box(-jnp.inf, jnp.inf, self.env.observation_shape)

    def init(self, key: Key) -> tuple[PgxState, Timestep]:
        pgx_state = self.env.init(key)
        env_state = PgxState(pgx_state=pgx_state)
        timestep = Timestep(
            next_obs=pgx_state.observation,
            true_next_obs=pgx_state.observation,
            action=self.action_space.sample(key),
            reward=pgx_state.rewards[pgx_state.current_player],
            terminated=pgx_state.terminated,
            truncated=pgx_state.truncated,
            info={},
        )
        return env_state, timestep

    def reset(self, key: Key, env_state: PgxState) -> tuple[PgxState, Timestep]:
        del env_state
        return self.init(key)

    def step(self, key: Key, env_state: PgxState, action: Array) -> tuple[PgxState, Timestep]:
        pgx_state = auto_reset(self.env.step, self.env.init)(env_state.pgx_state, action, key)
        env_state = PgxState(pgx_state=pgx_state)
        timestep = Timestep(
            next_obs=pgx_state.observation,
            true_next_obs=pgx_state.observation,
            action=action,
            reward=pgx_state.rewards[pgx_state.current_player],
            terminated=pgx_state.terminated,
            truncated=pgx_state.truncated,
            info={},
        )
        return env_state, timestep

    @property
    def num_actions(self) -> int:
        return self.env.num_actions


def make(id: str, **kwargs) -> Environment:
    return PgxAdapter(id, **kwargs)
