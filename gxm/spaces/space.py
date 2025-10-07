from typing import Any, Tuple

from jax import Array

Shape = Tuple[int, ...]


class Space:
    """Abstract base class for action and observation spaces."""

    def sample(self, key: Array, shape: Shape = ()) -> Any:
        del key, shape
        raise NotImplementedError

    def contains(self, x: Array) -> Any:
        del x
        raise NotImplementedError

    @property
    def n(self) -> int:
        raise NotImplementedError
