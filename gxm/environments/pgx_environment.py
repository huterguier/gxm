import jax
import pgx
from pgx.experimental import auto_reset

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Discrete


class PgxEnvironment(Environment):
    """Base class for Pgx environments."""

    env: pgx.Env

    def __init__(self, env_id: pgx.EnvId, **kwargs):
        self.env = pgx.make(env_id, **kwargs)
        self.action_space = Discrete(self.env.num_actions)

    def init(self, key: jax.Array) -> tuple[EnvironmentState, Timestep]:
        pgx_state = self.env.init(key)
        env_state = pgx_state
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
        self, key: jax.Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        del env_state
        return self.init(key)

    def step(
        self, key: jax.Array, env_state: EnvironmentState, action: jax.Array
    ) -> tuple[EnvironmentState, Timestep]:
        pgx_state = auto_reset(self.env.step, self.env.init)(
            env_state.state, action, key
        )
        env_state = pgx_state
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
