from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax

from gxm.core import Environment, EnvironmentState


@jax.tree_util.register_dataclass
@dataclass
class WrapperState(EnvironmentState):
    env_state: EnvironmentState


TWrapperState = TypeVar("TWrapperState", bound=WrapperState)


class Wrapper(Generic[TWrapperState], Environment[TWrapperState]):
    """Base class for environment wrappers in gxm."""

    env: Environment
    unwrap: bool = True

    def __init__(self, env: Environment, unwrap: bool = True):
        self.env = env
        if isinstance(env, Wrapper) and not unwrap:
            assert not env.unwrap
        self.unwrap = unwrap

    def has_wrapper(self, wrapper_type: type[Environment]) -> bool:
        if isinstance(self, wrapper_type):
            return True
        return self.env.has_wrapper(wrapper_type)

    def get_wrapper(self, wrapper_type: type[Environment]) -> Environment:
        if isinstance(self, wrapper_type):
            return self
        return self.env.get_wrapper(wrapper_type)

    @property
    def unwrapped(self) -> Environment:
        if self.unwrap:
            return self.env.unwrapped
        return self

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.env, name):
            return getattr(self.env, name)
        return getattr(self.env, name)
