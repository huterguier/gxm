from dataclasses import dataclass

import brax
import brax.envs
import jax
import jax.numpy as jnp

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box
from gxm.typing import Array, Key
from gxm.wrappers import AutoReset


@jax.tree_util.register_dataclass
@dataclass
class BraxState(EnvironmentState):
    brax_state: brax.envs.State


class BraxAdapter(Environment[BraxState]):
    """
    Adapter over a brax environment.

    Brax is created with ``auto_reset=False`` (its own auto-reset wrapper
    destroys the terminal observation), so this adapter does *not* auto-reset —
    ``make`` wraps it in :class:`gxm.wrappers.AutoReset` by default.

    Truncation is taken from brax's ``EpisodeWrapper``, which reports it in
    ``state.info["truncation"]``. Brax distinguishes a natural termination on
    the exact final step from a pure time-out, and its labeling (terminated
    wins the coincidence) is passed through unchanged.
    """

    brax_id: str
    env: brax.envs.Env

    def __init__(
        self, id: str, episode_length: int = 1000, action_repeat: int = 1, **kwargs
    ):
        self.brax_id = id
        self.id = f"Brax/{id}"
        self.env = brax.envs.create(
            id,
            episode_length=episode_length,
            action_repeat=action_repeat,
            auto_reset=False,
            **kwargs,
        )
        self.action_space = Box(
            low=jnp.full((self.env.action_size,), -1.0),
            high=jnp.full((self.env.action_size,), 1.0),
            shape=(self.env.action_size,),
        )
        self.observation_space = Box(
            low=jnp.full((self.env.observation_size,), -jnp.inf),
            high=jnp.full((self.env.observation_size,), jnp.inf),
            shape=(self.env.observation_size,),
        )

    def init(self, key: Key) -> tuple[BraxState, Timestep]:
        brax_state = self.env.reset(key)
        state = BraxState(brax_state=brax_state)
        timestep = Timestep(
            next_obs=brax_state.obs,
            true_next_obs=brax_state.obs,
            action=jax.tree.map(jnp.zeros_like, self.action_space.sample(key)),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=dict(brax_state.info),
        )
        return state, timestep

    def reset(self, key: Key, state: BraxState) -> tuple[BraxState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: BraxState, action: Array
    ) -> tuple[BraxState, Timestep]:
        del key
        brax_state = self.env.step(state.brax_state, action)
        state = BraxState(brax_state=brax_state)
        done = brax_state.done > 0.5
        truncated = brax_state.info["truncation"] > 0.5
        timestep = Timestep(
            next_obs=brax_state.obs,
            true_next_obs=brax_state.obs,
            action=action,
            reward=brax_state.reward,
            terminated=jnp.logical_and(done, jnp.logical_not(truncated)),
            truncated=truncated,
            info=dict(brax_state.info),
        )
        return state, timestep


def make(id: str, autoreset: bool = True, **kwargs) -> Environment:
    env = BraxAdapter(id, **kwargs)
    return AutoReset(env).seal() if autoreset else env
