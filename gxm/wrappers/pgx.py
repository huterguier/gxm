import jax
import pgx
from pgx.experimental import auto_reset

from gxm.core import Environment, EnvironmentState


class PgxEnvironment(Environment):
    """Base class for Gymnax environments."""

    env: pgx.Env

    def __init__(self, env_id: pgx.EnvId, **kwargs):
        self.env = pgx.make(env_id, **kwargs)

    def init(self, key: jax.Array) -> EnvironmentState:
        return self.reset(key)

    def reset(self, key: jax.Array) -> EnvironmentState:
        state = self.env.init(key)
        env_state = EnvironmentState(
            state=state,
            obs=state.observation,
            reward=state.rewards[state.current_player],
            done=state.terminated or state.truncated,
            info={},
        )
        return env_state

    def step(
        self, key: jax.Array, env_state: EnvironmentState, action: jax.Array
    ) -> EnvironmentState:
        state = auto_reset(self.env.step, self.env.init)(env_state.state, action, key)
        env_state = EnvironmentState(
            state=state,
            obs=state.observation,
            reward=state.rewards[state.current_player],
            done=state.terminated,
            info={},
        )
        return env_state

    @property
    def num_actions(self) -> int:
        return self.env.num_actions
