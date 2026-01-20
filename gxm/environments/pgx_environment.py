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
class PgxEnvironmentState(EnvironmentState):
    """State for Pgx environments."""

    pgx_state: pgx.State
    """The Pgx state."""


class PgxEnvironment(Environment[PgxEnvironmentState]):
    """Base class for Pgx environments."""

    pgx_id: str
    """The Pgx environment ID."""
    env: pgx.Env
    """The Pgx environment instance."""

    def __init__(self, id: str, **kwargs):
        self.id = id
        self.pgx_id = id.split("/", 1)[1]
        self.env = pgx.make(cast(pgx.EnvId, self.pgx_id), **kwargs)
        self.action_space = Discrete(self.env.num_actions)
        self.observation_space = Box(-jnp.inf, jnp.inf, self.env.observation_shape)

    def init(self, key: Key) -> tuple[PgxEnvironmentState, Timestep]:
        pgx_state = self.env.init(key)
        env_state = PgxEnvironmentState(pgx_state=pgx_state)
        timestep = Timestep(
            obs=pgx_state.observation,
            true_obs=pgx_state.observation,
            reward=pgx_state.rewards[pgx_state.current_player],
            terminated=pgx_state.terminated,
            truncated=pgx_state.truncated,
            info={},
        )
        return env_state, timestep

    def reset(
        self, key: Key, env_state: PgxEnvironmentState
    ) -> tuple[PgxEnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: Key, env_state: PgxEnvironmentState, action: Array
    ) -> tuple[PgxEnvironmentState, Timestep]:
        pgx_state = auto_reset(self.env.step, self.env.init)(
            env_state.pgx_state, action, key
        )
        env_state = PgxEnvironmentState(pgx_state=pgx_state)
        timestep = Timestep(
            obs=pgx_state.observation,
            true_obs=pgx_state.observation,
            reward=pgx_state.rewards[pgx_state.current_player],
            terminated=pgx_state.terminated,
            truncated=pgx_state.truncated,
            info={},
        )
        return env_state, timestep

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
