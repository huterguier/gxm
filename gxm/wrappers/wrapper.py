from typing import Any

from gxm.core import Environment


class Wrapper(Environment):
    """Base class for environment wrappers in gxm."""

    env: Environment

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)
