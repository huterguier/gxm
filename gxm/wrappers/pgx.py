from dataclasses import dataclass

import jax
import pgx
from pgx.experimental import auto_reset

from gxm.core import Env, EnvState, State


@jax.tree_util.register_dataclass
@dataclass
class PgxState(State):
    state_pgx: pgx.State


class PgxEnv(Env[PgxState]):
    """Base class for Gymnax environments."""

    env: pgx.Env

    def __init__(self, env_id: pgx.EnvId, **kwargs):
        self.env = pgx.make(env_id, **kwargs)

    def step(
        self, key: jax.Array, state: EnvState | PgxState, action: jax.Array
    ) -> EnvState:
        state = state.state if isinstance(state, EnvState) else state
        state_pgx = auto_reset(self.env.step, self.env.init)(
            state.state_pgx, action, key
        )

        state = PgxState(time=state.time + 1, state_pgx=state_pgx)
        env_state = EnvState(
            state=state,
            obs=state_pgx.observation,
            reward=state_pgx.rewards[state_pgx.current_player],
            done=state_pgx.terminated,
            info={},
        )
        return env_state

    def reset(self, key: jax.Array) -> EnvState:
        state_pgx = self.env.init(key)
        state = PgxState(time=0, state_pgx=state_pgx)
        env_state = EnvState(
            state=state,
            obs=state_pgx.observation,
            reward=state_pgx.rewards[state_pgx.current_player],
            done=state_pgx.terminated or state_pgx.truncated,
            info={},
        )
        return env_state

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
