from jax import Array

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.wrappers.wrapper import Wrapper


class EpisodicLife(Wrapper):
    """
    A wrapper that makes losing a life in an environment (like Atari games) count as the end of an episode.
    It assumes that the environment's timestep info dictionary contains a "lives" key indicating the number of lives remaining.
    """

    env: Environment

    def __init__(self, env: Environment):
        """
        Args:
            env: The environment to wrap.
        """
        self.env = env

    def init(self, key: Array) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.init(key)
        return (env_state, timestep.info["lives"]), timestep

    def reset(
        self, key: Array, env_state: EnvironmentState
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.reset(key, env_state[0])
        lives = timestep.info["lives"]
        return (env_state, lives), env_state.timestep

    def step(
        self,
        key: Array,
        env_state: EnvironmentState,
        action: Array,
    ) -> tuple[EnvironmentState, Timestep]:
        env_state, timestep = self.env.step(key, env_state[0], action)
        prev_lives = env_state[1]
        lives = timestep.info["lives"]
        timestep.terminated = timestep.terminated or (lives < prev_lives and lives > 0)
        return (env_state, lives), timestep
