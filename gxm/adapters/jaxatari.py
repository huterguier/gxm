from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jaxatari
import jaxatari.core
import jaxatari.spaces
import jaxatari.wrappers
from jaxatari.wrappers import AtariWrapper, PixelObsWrapper

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key


@jax.tree_util.register_dataclass
@dataclass
class JAXAtariState(EnvironmentState):
    jaxatari_state: Any


class JAXAtariAdapter(Environment[JAXAtariState]):
    """
    Adapter over a jaxatari environment with the standard DQN-style pixel
    pipeline (sticky actions, episodic life, frame skip, 84x84 grayscale
    frame stack).

    jaxatari's ``PixelObsWrapper`` auto-resets internally on the same step
    the episode ends, with no way to opt out. The environment returned here
    is therefore *already* auto-resetting — ``make`` does not add
    :class:`gxm.wrappers.AutoReset` — and, unlike every other adapter,
    ``true_next_obs`` always equals ``next_obs``: the pre-reset terminal
    observation is destroyed inside jaxatari and cannot be recovered. Do not
    bootstrap across truncated episode ends with this adapter.

    Termination and truncation are distinguished natively (truncation comes
    from ``max_frames_per_episode``); if both coincide, the step is labeled
    terminated.

    Requires jaxatari's sprite assets (run ``install-sprites`` once).

    Args:
        id: The jaxatari game name, e.g. ``"pong"``.
        atari_kwargs: Overrides for :class:`jaxatari.wrappers.AtariWrapper`
            (sticky actions, episodic life, noop starts, frame cap).
        pixel_kwargs: Overrides for :class:`jaxatari.wrappers.PixelObsWrapper`
            (resizing, grayscale, frame stacking/skipping, reward clipping).
        **kwargs: Forwarded to ``jaxatari.core.make``.
    """

    jaxatari_id: str
    env: jaxatari.wrappers.JaxatariWrapper

    def __init__(
        self,
        id: str,
        atari_kwargs: dict | None = None,
        pixel_kwargs: dict | None = None,
        **kwargs,
    ):
        self.jaxatari_id = id
        self.id = f"JAXAtari/{id}"
        env = jaxatari.core.make(id, **kwargs)
        env = AtariWrapper(env, **(atari_kwargs or {}))
        pixel_defaults: dict[str, Any] = dict(do_pixel_resize=True, grayscale=True)
        self.env = PixelObsWrapper(env, **{**pixel_defaults, **(pixel_kwargs or {})})
        self.action_space = _jaxatari_to_gxm_space(self.env.action_space())
        self.observation_space = _jaxatari_to_gxm_space(self.env.observation_space())
        # Probe one step eagerly to learn the info structure, so init can
        # return a structurally identical, zeroed info.
        key_probe = jax.random.key(0)
        _, state_probe = self.env.reset(key_probe)
        *_, info_probe = self.env.step(state_probe, jnp.int32(0))
        self._init_info = jax.tree.map(jnp.zeros_like, info_probe)

    def init(self, key: Key) -> tuple[JAXAtariState, Timestep]:
        obs, jaxatari_state = self.env.reset(key)
        state = JAXAtariState(jaxatari_state=jaxatari_state)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=jax.tree.map(jnp.zeros_like, self.action_space.sample(key)),
            reward=jnp.float32(0.0),
            terminated=jnp.bool(True),
            truncated=jnp.bool(False),
            info=self._init_info,
        )
        return state, timestep

    def reset(self, key: Key, state: JAXAtariState) -> tuple[JAXAtariState, Timestep]:
        del state
        return self.init(key)

    def step(
        self, key: Key, state: JAXAtariState, action: Array
    ) -> tuple[JAXAtariState, Timestep]:
        del key
        obs, jaxatari_state, reward, terminated, truncated, info = self.env.step(
            state.jaxatari_state, action
        )
        state = JAXAtariState(jaxatari_state=jaxatari_state)
        terminated = jnp.asarray(terminated)
        timestep = Timestep(
            next_obs=obs,
            true_next_obs=obs,
            action=action,
            reward=jnp.float32(reward),
            terminated=terminated,
            truncated=jnp.logical_and(
                jnp.asarray(truncated), jnp.logical_not(terminated)
            ),
            info=info,
        )
        return state, timestep


def _jaxatari_to_gxm_space(jaxatari_space) -> Space:
    if isinstance(jaxatari_space, jaxatari.spaces.Discrete):
        return Discrete(jaxatari_space.n)
    if isinstance(jaxatari_space, jaxatari.spaces.Box):
        return Box(
            low=jaxatari_space.low, high=jaxatari_space.high, shape=jaxatari_space.shape
        )
    if isinstance(jaxatari_space, jaxatari.spaces.Dict):
        return Tree(
            {k: _jaxatari_to_gxm_space(v) for k, v in jaxatari_space.spaces.items()}
        )
    raise NotImplementedError(
        f"JAXAtari space type {type(jaxatari_space)} not supported."
    )


def make(id: str, autoreset: bool = True, **kwargs) -> Environment:
    if not autoreset:
        raise ValueError(
            "JAXAtari environments auto-reset internally (inside jaxatari's "
            "PixelObsWrapper) and cannot be created without auto-resetting."
        )
    return JAXAtariAdapter(id, **kwargs)
