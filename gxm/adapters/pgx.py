from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import pgx

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete
from gxm.typing import Array, Key
from gxm.wrappers import AutoReset


@jax.tree_util.register_dataclass
@dataclass
class PgxState(EnvironmentState):
    pgx_state: pgx.State


class PgxAdapter(Environment[PgxState]):
    """
    Adapter over a pgx environment.

    Uses pgx's core ``step``, which does not auto-reset, so this adapter does
    not either — stepping past ``done`` is a no-op with zero rewards in pgx,
    but should be considered undefined. ``make`` wraps it in
    :class:`gxm.wrappers.AutoReset` by default.

    pgx distinguishes termination and truncation natively; if both coincide,
    the step is labeled terminated. The reward is the acting player's reward,
    i.e. indexed by ``current_player`` *before* the step. The current player
    and the legal action mask are exposed through ``info`` — many pgx
    environments are games where sampling from the full action space takes
    illegal actions, which pgx punishes with immediate termination.
    """

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
        state = PgxState(pgx_state=pgx_state)
        timestep = Timestep(
            next_obs=pgx_state.observation,
            true_next_obs=pgx_state.observation,
            action=jax.tree.map(jnp.zeros_like, self.action_space.sample(key)),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info={
                "current_player": pgx_state.current_player,
                "legal_action_mask": pgx_state.legal_action_mask,
            },
        )
        return state, timestep

    def reset(self, key: Key, state: PgxState) -> tuple[PgxState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: PgxState, action: Array
    ) -> tuple[PgxState, Timestep]:
        player = state.pgx_state.current_player
        pgx_state = self.env.step(state.pgx_state, action, key)
        state = PgxState(pgx_state=pgx_state)
        terminated = pgx_state.terminated
        timestep = Timestep(
            next_obs=pgx_state.observation,
            true_next_obs=pgx_state.observation,
            action=action,
            reward=jnp.float32(pgx_state.rewards[player]),
            terminated=terminated,
            truncated=jnp.logical_and(
                pgx_state.truncated, jnp.logical_not(terminated)
            ),
            info={
                "current_player": pgx_state.current_player,
                "legal_action_mask": pgx_state.legal_action_mask,
            },
        )
        return state, timestep

    @property
    def num_actions(self) -> int:
        return self.env.num_actions


def make(id: str, autoreset: bool = True, **kwargs) -> Environment:
    env = PgxAdapter(id, **kwargs)
    return AutoReset(env).seal() if autoreset else env
