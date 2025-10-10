import jax
import pytest
from test_space import TestSpace

from gxm.spaces import Box


class TestBox(TestSpace):
    @pytest.fixture
    def space(self, low_high, shape):
        low, high = low_high
        return Box(low=low, high=high, shape=shape)

    @pytest.fixture(params=[(-1.0, 1.0), (0.0, 10.0), (-5.0, 5.0)])
    def low_high(self, request):
        return request.param

    @pytest.fixture(params=[(3,), (2, 2), (4, 3, 2)])
    def shape(self, request):
        return request.param

    def test_init(self, space):
        assert isinstance(space, Box)

    def test_broadcast(self, shape, low_high):
        low, high = low_high
        box = Box(low=low, high=high, shape=shape)
        assert box.low.shape == shape
        assert box.high.shape == shape

    def test_sample(self, space, key):
        sample = space.sample(key)
        assert sample.shape == space.low.shape
        assert jax.numpy.all(sample >= space.low)
        assert jax.numpy.all(sample <= space.high)

    def test_contains(self, space):
        valid_sample = (space.low + space.high) / 2
        invalid_sample_low = space.low - 1.0
        invalid_sample_high = space.high + 1.0
        assert space.contains(valid_sample)
        assert not space.contains(invalid_sample_low)
        assert not space.contains(invalid_sample_high)

    def test_sample_contains(self, space, key):
        sample = space.sample(key)
        assert space.contains(sample)

    def test_repr(self, space):
        assert (
            repr(space)
            == f"Box(low={space.low}, high={space.high}, shape={space.shape})"
        )
