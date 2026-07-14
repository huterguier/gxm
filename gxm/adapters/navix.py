from dataclasses import dataclass

import jax
import jax.numpy as jnp
import navix

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space
from gxm.typing import Action, Key
from gxm.wrappers import AutoReset


@jax.tree_util.register_dataclass
@dataclass
class NavixState(EnvironmentState):
    navix_state: navix.Timestep


class NavixAdapter(Environment[NavixState]):
    """
    Adapter over a navix environment.

    Uses navix's non-resetting ``_step`` rather than ``step``: navix's own
    ``step`` auto-resets *on the step after* a terminal one, ignoring the
    action passed to it — an episode-boundary sentinel that must not enter
    the transition stream. Consequently this adapter does *not* auto-reset —
    stepping past ``done`` is undefined. ``make`` wraps it in
    :class:`gxm.wrappers.AutoReset` by default.

    Termination and truncation are taken from navix's own ``StepType``, which
    distinguishes the two exactly.
    """

    navix_id: str
    env: navix.environments.environment.Environment

    def __init__(self, id: str, **kwargs):
        self.navix_id = id
        self.id = f"Navix/{id}"
        self.env = navix.make(id, **kwargs)
        self.action_space = _navix_to_gxm_space(self.env.action_space)
        self.observation_space = _navix_to_gxm_space(self.env.observation_space)
        # navix declares too-tight observation bounds for some environments
        # (e.g. Empty-5x5 declares a maximum of 8 but emits entity ids up to
        # 10). If a probe observation escapes the declared space, widen the
        # bound to the dtype's range.
        obs_probe = self.env.reset(jax.random.key(0)).observation
        if not bool(self.observation_space.contains(obs_probe)):
            self.observation_space = Box(
                low=0.0,
                high=float(jnp.iinfo(obs_probe.dtype).max),
                shape=jnp.shape(obs_probe),
            )

    def init(self, key: Key) -> tuple[NavixState, Timestep]:
        navix_state = self.env.reset(key)
        state = NavixState(navix_state=navix_state)
        timestep = Timestep(
            next_obs=navix_state.observation,
            true_next_obs=navix_state.observation,
            action=jax.tree.map(jnp.zeros_like, self.action_space.sample(key)),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=dict(navix_state.info),
        )
        return state, timestep

    def reset(self, key: Key, state: NavixState) -> tuple[NavixState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: NavixState, action: Action
    ) -> tuple[NavixState, Timestep]:
        del key
        navix_state = self.env._step(state.navix_state, action)
        state = NavixState(navix_state=navix_state)
        timestep = Timestep(
            next_obs=navix_state.observation,
            true_next_obs=navix_state.observation,
            action=action,
            reward=navix_state.reward,
            terminated=navix_state.is_termination(),
            truncated=navix_state.is_truncation(),
            info=dict(navix_state.info),
        )
        return state, timestep

    @property
    def num_actions(self) -> int:
        return len(self.env.action_set)


def _navix_to_gxm_space(navix_space) -> Space:
    if isinstance(navix_space, navix.spaces.Discrete):
        # navix Discrete spaces carry a shape (categorical grids); gxm's
        # Discrete is scalar-only, so shaped ones map to an integer Box.
        if tuple(navix_space.shape) == ():
            return Discrete(int(navix_space.n))
        return Box(
            low=navix_space.minimum,
            high=navix_space.maximum,
            shape=tuple(navix_space.shape),
        )
    if isinstance(navix_space, navix.spaces.Continuous):
        return Box(
            low=navix_space.minimum, high=navix_space.maximum, shape=navix_space.shape
        )
    raise NotImplementedError(f"Navix space type {type(navix_space)} not supported.")


def make(id: str, autoreset: bool = True, **kwargs) -> Environment:
    env = NavixAdapter(id, **kwargs)
    return AutoReset(env).seal() if autoreset else env
