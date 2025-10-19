from typing import Any

from gxm.core import Environment


class Wrapper(Environment):
    """Base class for environment wrappers in gxm."""

    env: Environment

    def has_wrapper(self, wrapper_type: type[Environment]) -> bool:
        if isinstance(self, wrapper_type):
            return True
        return self.env.has_wrapper(wrapper_type)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.env, name)
