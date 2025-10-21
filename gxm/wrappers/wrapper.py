from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax

from gxm.core import Environment, EnvironmentState


@jax.tree_util.register_dataclass
@dataclass
class WrapperState(EnvironmentState):
    env_state: EnvironmentState

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.env, name):
            return getattr(self.env, name)
        return getattr(self.env, name)


TWrapperState = TypeVar("TWrapperState", bound=EnvironmentState)


class Wrapper(Generic[TWrapperState], Environment[TWrapperState]):
    """Base class for environment wrappers in gxm."""

    env: Environment

    def has_wrapper(self, wrapper_type: type[Environment]) -> bool:
        if isinstance(self, wrapper_type):
            return True
        return self.env.has_wrapper(wrapper_type)

    def get_wrapper(self, wrapper_type: type[Environment]) -> Environment:
        if isinstance(self, wrapper_type):
            return self
        return self.env.get_wrapper(wrapper_type)

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.env, name):
            return getattr(self.env, name)
        return getattr(self.env, name)
