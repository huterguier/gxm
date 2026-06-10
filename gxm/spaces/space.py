from typing import Any

from gxm.typing import Array

Shape = tuple[int, ...]


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
