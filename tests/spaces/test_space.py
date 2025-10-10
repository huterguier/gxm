import jax
import pytest

from gxm.core import Environment
from gxm.spaces import Space


class TestSpace:
    @pytest.fixture(params=[])
    def space(request) -> Space:
        raise NotImplementedError("Add environments to the fixture parameters.")

    @pytest.fixture(params=[0, 1, 42])
    def key(self, request):
        return jax.random.key(request.param)

    def test_init(self, space):
        del space
        raise NotImplementedError("Implement test_init method in subclass.")

    def test_sample(self, space, key):
        del space, key
        raise NotImplementedError("Implement test_sample method in subclass.")

    def test_contains(self, space):
        del space
        raise NotImplementedError("Implement test_contains method in subclass.")

    def test_sample_contains(self, space, key):
        sample = space.sample(key)
        assert space.contains(sample)

    def test_repr(self, space):
        del space
        raise NotImplementedError("Implement test_repr method in subclass.")
