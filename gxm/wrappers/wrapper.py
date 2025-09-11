from typing import Any

from gxm.core import Environment


class Wrapper(Environment):
    """Base class for environment wrappers in gxm."""

    env: Environment

    @property
    def num_actions(self) -> int:
        return self.env.num_actions

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)
