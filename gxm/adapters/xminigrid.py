from dataclasses import dataclass

import jax
import jax.numpy as jnp
import xminigrid
import xminigrid.environment
from xminigrid.core.constants import NUM_COLORS, NUM_TILES

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete
from gxm.typing import Array, Key
from gxm.wrappers import AutoReset


@jax.tree_util.register_dataclass
@dataclass
class XMiniGridState(EnvironmentState):
    xminigrid_state: xminigrid.environment.TimeStep


class XMiniGridAdapter(Environment[XMiniGridState]):
    """
    Adapter over an xminigrid environment.

    xminigrid does not auto-reset, so this adapter does not either — stepping
    past ``done`` is undefined. ``make`` wraps it in
    :class:`gxm.wrappers.AutoReset` by default.

    xminigrid reports episode ends dm_env-style through ``step_type`` and
    ``discount``: ``discount`` is zero exactly when the goal was reached.
    A last step is therefore labeled terminated when ``discount == 0`` and
    truncated otherwise. Reaching the goal on the very step the time limit
    expires yields ``discount == 0`` and is labeled terminated — here the
    labeling is exact, because xminigrid computes the two conditions
    separately.
    """

    xminigrid_id: str
    env: xminigrid.environment.Environment
    env_params: xminigrid.environment.EnvParams

    def __init__(self, id: str, **kwargs):
        self.xminigrid_id = id
        self.id = f"XMiniGrid/{id}"
        self.env, self.env_params = xminigrid.make(id, **kwargs)
        self.action_space = Discrete(self.env.num_actions(self.env_params))
        observation_shape = self.env.observation_shape(self.env_params)
        assert type(observation_shape) is tuple
        # Observations are grids of (tile id, color id) pairs on the last axis.
        if observation_shape[-1] == 2:
            high = jnp.asarray([NUM_TILES - 1, NUM_COLORS - 1], dtype=jnp.float32)
        else:
            high = jnp.float32(max(NUM_TILES, NUM_COLORS) - 1)
        self.observation_space = Box(low=0.0, high=high, shape=observation_shape)

    def init(self, key: Key) -> tuple[XMiniGridState, Timestep]:
        xminigrid_state = self.env.reset(self.env_params, key)
        state = XMiniGridState(xminigrid_state=xminigrid_state)
        timestep = Timestep(
            next_obs=xminigrid_state.observation,
            true_next_obs=xminigrid_state.observation,
            action=jax.tree.map(jnp.zeros_like, self.action_space.sample(key)),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info={},
        )
        return state, timestep

    def reset(self, key: Key, state: XMiniGridState) -> tuple[XMiniGridState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: XMiniGridState, action: Array
    ) -> tuple[XMiniGridState, Timestep]:
        del key
        xminigrid_state = self.env.step(self.env_params, state.xminigrid_state, action)
        state = XMiniGridState(xminigrid_state=xminigrid_state)
        last = xminigrid_state.last()
        terminated = jnp.logical_and(last, xminigrid_state.discount == 0.0)
        timestep = Timestep(
            next_obs=xminigrid_state.observation,
            true_next_obs=xminigrid_state.observation,
            action=action,
            reward=jnp.float32(xminigrid_state.reward),
            terminated=terminated,
            truncated=jnp.logical_and(last, jnp.logical_not(terminated)),
            info={},
        )
        return state, timestep


def make(id: str, autoreset: bool = True, **kwargs) -> Environment:
    env = XMiniGridAdapter(id, **kwargs)
    return AutoReset(env).seal() if autoreset else env
