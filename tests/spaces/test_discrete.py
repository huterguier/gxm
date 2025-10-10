import pytest
from test_space import TestSpace

from gxm.spaces import Discrete


class TestDiscrete(TestSpace):
    @pytest.fixture(params=[1, 3, 8])
    def space(self, request) -> Discrete:
        return Discrete(request.param)

    def test_init(self, space):
        assert isinstance(space, Discrete)
        assert space.n > 0

    def test_sample(self, space, key):
        sample = space.sample(key)
        assert sample.shape == ()
        assert 0 <= sample < space.n

    def test_contains(self, space):
        for i in range(space.n):
            assert space.contains(i)
        assert not space.contains(-1)
        assert not space.contains(space.n)

    def test_repr(self, space):
        assert repr(space) == f"Discrete({space.n})"
