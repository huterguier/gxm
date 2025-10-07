from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .space import Space

Shape = Tuple[int, ...]


class Discrete(Space):
    """A discrete space consiting of :math:`\{0, 1, ..., n-1\}`."""

    _n: int

    def __init__(self, n: int):
        assert n >= 0
        self._n = n

    def sample(self, key: Array, shape: Shape = ()) -> Array:
        """
        Sample random action uniformly from :math:`\{0, 1, ..., n-1\}`.

        Arguments:
            key: A JAX random key used for sampling.
            shape: The shape of the returned sample.
        Returns:
            The sampled value.
        """
        return jax.random.randint(key, shape, 0, self.n)

    def contains(self, x: Any) -> Array:
        """
        Check whether specific object is within space.

        Arguments:
            x: The object to be checked.
        Returns:
            Whether the object is contained in the space.
        """
        return jnp.logical_and(x >= 0, x < self.n)

    @property
    def n(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"
