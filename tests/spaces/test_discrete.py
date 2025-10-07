import jax
import pytest

from gxm.spaces import Discrete


class TestDiscrete:
    @pytest.fixture(params=[1, 3, 8])
    def space(self, request):
        return Discrete(request.param)

    @pytest.fixture(params=[0, 1, 42])
    def key(self, request):
        return jax.random.key(request.param)

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

    def test_sample_contains(self, space, key):
        sample = space.sample(key)
        assert space.contains(sample)

    def test_repr(self, space):
        assert repr(space) == f"Discrete({space.n})"
