import jax
import pytest
from test_space import TestSpace

from gxm.spaces import Box, Discrete, Tree


class TestDiscrete(TestSpace):

    @pytest.fixture(
        params=[
            Tree((Discrete(2), Discrete(3))),
            Tree([Discrete(3), Discrete(4)]),
            Tree({"a": Discrete(2), "b": Discrete(3)}),
            Tree((Discrete(2), Box(-1.0, 1.0, (4,)))),
            Tree((Box(-1.0, 1.0, (4,)), Discrete(3))),
            Tree((Discrete(2), [Discrete(2), Discrete(3)])),
        ]
    )
    def space(self, request) -> Discrete:
        return request.param

    def test_init(self, space):
        assert isinstance(space, Tree)

    def test_sample(self, space, key):
        sample = space.sample(key)
        assert jax.tree.structure(sample) == jax.tree.structure(space.spaces)

    def test_contains(self, space):
        pass

    def test_n(self, space):
        if all(isinstance(s, Discrete) for s in jax.tree.leaves(space.spaces)):
            print("Testing n...")
            expected_n = 1
            for s in jax.tree.leaves(space.spaces):
                expected_n *= s.n
            print(expected_n)
            assert space.n == expected_n

    def test_repr(self, space):
        assert repr(space) == f"Tree({space.spaces})"
