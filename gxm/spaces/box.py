from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .space import Space

Shape = Tuple[int, ...]


class Box(Space):
    r"""A bounded box in :math:`\mathbb{R}^n`."""

    low: Array
    """Lower bound of the box."""
    high: Array
    """Upper bound of the box."""
    shape: Shape
    """Shape of the box."""

    def __init__(
        self,
        low: Array | float,
        high: Array | float,
        shape: Shape,
    ):
        self.low = jnp.broadcast_to(jnp.asarray(low, dtype=jnp.float32), shape)
        self.high = jnp.broadcast_to(jnp.asarray(high, dtype=jnp.float32), shape)
        self.shape = shape

    def sample(self, key: Array, shape: Shape = ()) -> Any:
        """
        Sample uniformly from the box.

        Args:
            key: JAX random key.
            shape: Shape of the sample to be drawn.
        Returns:
            Sample drawn from the box.
        """
        return jax.random.uniform(
            key, shape + self.shape, minval=self.low, maxval=self.high
        )

    def contains(self, x: Any) -> Array:
        """
        Check whether specific object is within space.

        Args:
            x: Object to be checked.
        Returns:
            Whether the object is within the space.
        """
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond

    @property
    def n(self) -> int:
        raise NotImplementedError("Box space does not have a single dimension 'n'.")

    def __repr__(self) -> str:
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"
