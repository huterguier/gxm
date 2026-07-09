from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax

from gxm.core import Dynamics, Environment, EnvironmentState, Timestep, TStep


@jax.tree_util.register_dataclass
@dataclass
class WrapperState(EnvironmentState):
    wrapped_state: EnvironmentState


TWrapperState = TypeVar("TWrapperState", bound=WrapperState)


class Wrapper(Generic[TWrapperState, TStep], Dynamics[TWrapperState, TStep]):
    """Base class for wrappers in gxm, over either bare Dynamics or an Environment."""

    wrapped: Dynamics[Any, TStep]
    unwrap: bool = True

    def __init__(self, wrapped: Dynamics[Any, TStep], unwrap: bool = True):
        self.wrapped = wrapped
        self.id = wrapped.id
        self.action_space = wrapped.action_space
        self.observation_space = wrapped.observation_space
        if isinstance(wrapped, Wrapper) and not unwrap:
            assert not wrapped.unwrap
        self.unwrap = unwrap

    def has_wrapper(self, wrapper_type: type[Dynamics]) -> bool:
        if isinstance(self, wrapper_type):
            return True
        return self.wrapped.has_wrapper(wrapper_type)

    def get_wrapper(self, wrapper_type: type[Dynamics]) -> Dynamics:
        if isinstance(self, wrapper_type):
            return self
        return self.wrapped.get_wrapper(wrapper_type)

    @property
    def unwrapped(self) -> Dynamics:
        if self.unwrap:
            return self.wrapped.unwrapped
        return self

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.wrapped, name):
            return getattr(self.wrapped, name)
        raise AttributeError(name)


class EnvironmentWrapper(Generic[TWrapperState], Wrapper[TWrapperState, Timestep]):
    """Base class for wrappers that only operate on Environments (need reward/terminated/truncated)."""

    wrapped: Environment

    def __init__(self, wrapped: Environment, unwrap: bool = True):
        super().__init__(wrapped, unwrap=unwrap)
