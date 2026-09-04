import hashlib
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np

from gxm.core import Environment, EnvironmentState, Timestep
from gxm.spaces import Box, Discrete, Space, Tree
from gxm.typing import Array, Key

_SEED_MODULUS = 1 << 30
"""Kept inside ``int32``: Gymnasium spreads a scalar seed as ``seed + i``, ALE narrows it."""

_LIVE_ENV_WARN_THRESHOLD = 32


def _seed_from_key_data(key_data: Any) -> int:
    digest = hashlib.blake2b(np.asarray(key_data).tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _SEED_MODULUS


def seed_from_key(key: Key) -> int:
    """
    The Gymnasium seed the adapter derives from ``key``.

    A scalar is the only seed form every Gymnasium vector env accepts, so the whole
    batch of keys is hashed down to one integer, which Gymnasium then spreads over
    sub-envs as ``seed + i``. The batch is reproducible from the batch of keys;
    individual sub-envs are not seeded from their own key.
    """
    return _seed_from_key_data(jax.random.key_data(key))


@jax.tree_util.register_dataclass
@dataclass
class GymnasiumState(EnvironmentState):
    env_id: Array


class GymnasiumAdapter(Environment[GymnasiumState]):
    """
    Adapter over a Gymnasium vector environment.

    The batch size is only known inside :meth:`init`, so the adapter holds a factory
    ``num_envs -> gymnasium.VectorEnv`` rather than an env object. The factory must be
    callable repeatedly, return a vector env whose ``num_envs`` matches its argument,
    and produce per-env shapes independent of ``num_envs``.
    """

    factory: Callable[[int], Any]
    return_shape_dtype: Any

    def __init__(self, factory: Callable[[int], Any], id: str = "Gymnasium/wrapped"):
        self.factory = factory
        self.id = id
        self._envs: dict[int, Any] = {}
        self._next_env_id = 0
        env = self._make_vec(1)
        obs, info_reset = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info_step = env.step(action)
        if jax.tree.structure(info_reset) != jax.tree.structure(info_step):
            info = info_step
        else:
            info = info_reset
        state = GymnasiumState(env_id=jnp.int32(0))
        timestep = Timestep(
            next_obs=jnp.array(obs),
            true_next_obs=jnp.array(obs),
            action=jnp.array(action),
            reward=jnp.array(reward, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool),
            truncated=jnp.array(truncated, dtype=jnp.bool),
            info=jax.tree.map(jnp.array, info),
        )
        self.return_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), (state, timestep)
        )
        self.action_space = _gymnasium_to_gxm_space(env.single_action_space)
        self.observation_space = _gymnasium_to_gxm_space(env.single_observation_space)
        env.close()

    def _make_vec(self, num_envs: int) -> Any:
        envs = self.factory(num_envs)
        if envs.num_envs != num_envs:
            raise ValueError(
                f"Environment factory for '{self.id}' returned a vector env with "
                f"num_envs={envs.num_envs}, expected {num_envs}."
            )
        return envs

    def _lookup_envs(self, env_id: Any) -> Any:
        flat = np.ravel(env_id)
        first = int(flat[0])
        if not np.all(flat == first):
            raise ValueError(
                f"State for '{self.id}' mixes env ids {np.unique(flat).tolist()}; states "
                f"from separate init calls cannot be spliced or stacked."
            )
        envs = self._envs.get(first)
        if envs is None:
            raise ValueError(
                f"State for '{self.id}' refers to env id {first}, which is not live: it "
                f"was closed, or created by a different adapter."
            )
        if envs.num_envs != flat.size:
            raise ValueError(
                f"State for '{self.id}' has batch size {flat.size}, but env id {first} "
                f"was created with num_envs={envs.num_envs}."
            )
        return envs

    def close(self):
        for envs in self._envs.values():
            envs.close()
        self._envs.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def init(self, key: Key) -> tuple[GymnasiumState, Timestep]:
        def callback(key):
            shape = key.shape[:-1]
            keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]
            envs = self._make_vec(num_envs)
            obs, info = envs.reset(seed=_seed_from_key_data(keys_flat))
            if jax.tree.structure(info) != jax.tree.structure(
                self.return_shape_dtype[1].info
            ):
                info = jax.tree.map(
                    lambda x: jnp.zeros((num_envs,) + x.shape[1:], x.dtype),
                    self.return_shape_dtype[1].info,
                )
            env_id = self._next_env_id
            self._next_env_id += 1
            self._envs[env_id] = envs
            if len(self._envs) == _LIVE_ENV_WARN_THRESHOLD:
                warnings.warn(
                    f"'{self.id}' holds {len(self._envs)} live host environments; every "
                    f"init creates another. Use reset to reinitialize an existing state, "
                    f"or close() to release them.",
                    stacklevel=2,
                )
            action_spec = self.return_shape_dtype[1].action
            action_sentinel = jnp.zeros(
                (num_envs,) + action_spec.shape, action_spec.dtype
            )
            state = GymnasiumState(env_id=jnp.full(shape, env_id, dtype=jnp.int32))
            timestep = Timestep(
                next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                action=jnp.reshape(action_sentinel, shape + action_spec.shape),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.ones(shape, dtype=jnp.bool),
                truncated=jnp.zeros(shape, dtype=jnp.bool),
                info=jax.tree.map(lambda i: jnp.reshape(i, shape + i.shape[1:]), info),
            )
            return state, timestep

        state, timestep = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            jax.random.key_data(key),
            vmap_method="broadcast_all",
        )
        return state, timestep

    def reset(self, key: Key, state: GymnasiumState) -> tuple[GymnasiumState, Timestep]:
        def callback(key, env_id):
            shape = env_id.shape
            envs = self._lookup_envs(env_id)
            num_envs = env_id.size
            obs, info = envs.reset(seed=_seed_from_key_data(key))
            if jax.tree.structure(info) != jax.tree.structure(
                self.return_shape_dtype[1].info
            ):
                info = jax.tree.map(
                    lambda x: jnp.zeros((num_envs,) + x.shape[1:], x.dtype),
                    self.return_shape_dtype[1].info,
                )
            action_spec = self.return_shape_dtype[1].action
            action_sentinel = jnp.zeros(
                (num_envs,) + action_spec.shape, action_spec.dtype
            )
            state = GymnasiumState(env_id=jnp.full(shape, env_id, dtype=jnp.int32))
            timestep = Timestep(
                next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                action=jnp.reshape(action_sentinel, shape + action_spec.shape),
                reward=jnp.zeros(shape, dtype=jnp.float32),
                terminated=jnp.ones(shape, dtype=jnp.bool),
                truncated=jnp.zeros(shape, dtype=jnp.bool),
                info=jax.tree.map(lambda i: jnp.reshape(i, shape + i.shape[1:]), info),
            )
            return state, timestep

        state, timestep = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            jax.random.key_data(key),
            state.env_id,
            vmap_method="broadcast_all",
        )
        return state, timestep

    def step(
        self, key: Key, state: GymnasiumState, action: Array
    ) -> tuple[GymnasiumState, Timestep]:
        del key

        def callback(env_id, action):
            shape = env_id.shape
            envs = self._lookup_envs(env_id)
            actions = jax.tree.map(
                lambda a: np.asarray(a).reshape(-1, *action.shape[len(shape) :]), action
            )
            obs, reward, terminated, truncated, info = envs.step(actions)
            state = GymnasiumState(env_id=env_id)
            timestep = Timestep(
                next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                true_next_obs=jnp.reshape(obs, shape + obs.shape[1:]),
                action=jnp.reshape(action, shape + action.shape[len(shape) :]),
                reward=jnp.reshape(reward, shape).astype(jnp.float32),
                terminated=jnp.reshape(terminated, shape).astype(jnp.bool),
                truncated=jnp.reshape(truncated, shape).astype(jnp.bool),
                info=jax.tree.map(lambda i: jnp.reshape(i, shape + i.shape[1:]), info),
            )
            return state, timestep

        state, timestep = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            state.env_id,
            action,
            vmap_method="broadcast_all",
        )
        return state, timestep


def _gymnasium_to_gxm_space(gymnasium_space: Any) -> Space:
    if isinstance(gymnasium_space, gymnasium.spaces.Discrete):
        return Discrete(int(gymnasium_space.n))
    elif isinstance(gymnasium_space, gymnasium.spaces.Box):
        return Box(
            jnp.asarray(gymnasium_space.low),
            jnp.asarray(gymnasium_space.high),
            gymnasium_space.shape,
        )
    elif isinstance(gymnasium_space, gymnasium.spaces.MultiDiscrete):
        return Tree(tuple(Discrete(int(n)) for n in gymnasium_space.nvec))
    elif isinstance(gymnasium_space, gymnasium.spaces.Dict):
        return Tree(
            {k: _gymnasium_to_gxm_space(v) for k, v in gymnasium_space.spaces.items()}
        )
    elif isinstance(gymnasium_space, gymnasium.spaces.Tuple):
        return Tree(tuple(_gymnasium_to_gxm_space(s) for s in gymnasium_space.spaces))
    else:
        raise NotImplementedError(f"Gymnasium space {gymnasium_space} not supported.")


def make(id: str, **kwargs) -> Environment:
    return GymnasiumAdapter(
        lambda num_envs: gymnasium.make_vec(id, num_envs=num_envs, **kwargs),
        id=f"Gymnasium/{id}",
    )


def wrap(factory: Callable[[int], Any], id: str = "Gymnasium/wrapped") -> Environment:
    """
    Wrap a Gymnasium vector-environment factory as a gxm environment.

    Takes a factory rather than an env object because a :class:`gymnasium.VectorEnv`
    has its batch size baked in while a gxm environment does not. Use it for envs
    outside Gymnasium's registry, or setups :func:`gymnasium.make_vec` cannot express
    (vector-level wrappers, custom ``VectorEnv`` implementations).
    """
    return GymnasiumAdapter(factory, id=id)
