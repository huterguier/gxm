import jax
import pytest

import gxm
from gxm.core import Environment
from gxm.wrappers import Wrapper


class TestWrapper:
    __test__ = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__test__ = True

    @pytest.fixture(
        params=[
            "Gymnax/CartPole-v1",
            "Gymnax/Breakout-MinAtar",
        ]
    )
    def env(self, request) -> Environment:
        return gxm.make(request.param)

    @pytest.fixture
    def wrapper(self, env) -> Wrapper:
        pytest.skip("Base Wrapper class cannot be instantiated directly.")

    def test_init(self, wrapper: Wrapper):
        key = jax.random.key(0)
        wrapper_state, timestep = wrapper.init(key)

    def test_reset(self, wrapper: Wrapper):
        key = jax.random.key(0)
        wrapper_state, timestep = wrapper.init(key)
        wrapper_state, timestep = wrapper.reset(key, wrapper_state)

    def test_step(self, wrapper: Wrapper):
        key = jax.random.key(0)
        wrapper_state, timestep = wrapper.init(key)
        action = wrapper.action_space.sample(key)
        wrapper_state, timestep = wrapper.step(key, wrapper_state, action)

    def test_vmap_init(self, wrapper: Wrapper):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        wrapper_states, timesteps = jax.vmap(wrapper.init)(keys)

    def test_vmap_reset(self, wrapper: Wrapper):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        wrapper_states, timesteps = jax.vmap(wrapper.init)(keys)
        wrapper_states, timesteps = jax.vmap(wrapper.reset)(keys, wrapper_states)

    def test_vmap_step(self, wrapper: Wrapper):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        wrapper_states, timesteps = jax.vmap(wrapper.init)(keys)
        actions = wrapper.action_space.sample(key, (10,))
        wrapper_states, timesteps = jax.vmap(wrapper.step)(
            keys, wrapper_states, actions
        )

    def test_scan_step(self, wrapper: Wrapper):
        key = jax.random.key(0)
        wrapper_state, _ = wrapper.init(key)

        def step_fn(carry, _):
            wrapper_state = carry
            action = wrapper.action_space.sample(key)
            wrapper_state, timestep = wrapper.step(key, wrapper_state, action)
            return wrapper_state, timestep

        jax.lax.scan(step_fn, wrapper_state, length=10)


def test_seal():
    """seal() returns a sealed copy: unwrapped stops at the sealed stack,
    wrappers added on top stay removable, and the original is not mutated."""
    from gxm.wrappers import AutoReset, ClipReward

    raw = gxm.make("Gymnax/CartPole-v1", autoreset=False)
    stack = AutoReset(raw)
    sealed = stack.seal()

    # Functional: the original stack is untouched and still peels fully.
    assert stack.unwrap
    assert stack.unwrapped is raw
    # Sealed copy: unwrapped stops at the sealed layer.
    assert not sealed.unwrap
    assert sealed.unwrapped is sealed
    assert sealed.wrapped is raw

    # Wrappers added on top of a sealed stack remain removable.
    outer = ClipReward(sealed)
    assert outer.unwrap
    assert outer.unwrapped is sealed

    # Introspection sees through sealed layers.
    assert outer.has_wrapper(AutoReset)
    assert outer.get_wrapper(AutoReset) is sealed


def test_seal_marks_whole_stack():
    """Sealing marks every wrapper below the seal point, so the sealed region
    is always contiguous down to the base environment."""
    from gxm.wrappers import AutoReset, ClipReward

    raw = gxm.make("Gymnax/CartPole-v1", autoreset=False)
    sealed = ClipReward(AutoReset(raw)).seal()
    assert not sealed.unwrap
    assert not sealed.wrapped.unwrap
    assert sealed.unwrapped is sealed


def test_getattr_without_wrapped_does_not_recurse():
    """Attribute access on a Wrapper whose __init__ never ran (as happens
    during unpickling/copy) must raise AttributeError, not RecursionError."""
    from gxm.wrappers import AutoReset

    wrapper = object.__new__(AutoReset)
    with pytest.raises(AttributeError):
        _ = wrapper.action_space
    with pytest.raises(AttributeError):
        _ = wrapper.wrapped


def test_wrapper_copy():
    import copy

    from gxm.wrappers import AutoReset

    env = gxm.make("Gymnax/CartPole-v1", autoreset=False)
    wrapper = AutoReset(env)
    clone = copy.copy(wrapper)
    assert clone.wrapped is env
    clone = copy.deepcopy(wrapper)
    key = jax.random.key(0)
    state, timestep = clone.init(key)
    assert timestep is not None
