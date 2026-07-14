import copy
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
    """Base class for wrappers in gxm, over either bare Dynamics or an Environment.

    Wrappers are *removable* by default: ``unwrapped`` peels them off to reach
    the base environment. A wrapper stack can instead be marked as part of the
    environment's definition by calling :meth:`seal` — e.g. Atari-style
    preprocessing, or the auto-reset layer added by ``gxm.make`` — in which
    case ``unwrapped`` stops peeling there. Introspection is unaffected:
    ``has_wrapper`` and ``get_wrapper`` see through sealed wrappers.
    """

    wrapped: Dynamics[Any, TStep]
    unwrap: bool = True

    def __init__(self, wrapped: Dynamics[Any, TStep]):
        self.wrapped = wrapped
        self.id = wrapped.id
        self.action_space = wrapped.action_space
        self.observation_space = wrapped.observation_space

    def seal(self) -> "Wrapper":
        """
        Return a copy of this wrapper stack marked as part of the environment
        definition: ``unwrapped`` will not peel past it. Wrappers added on top
        of the sealed stack remain removable as usual.

        Does not mutate ``self``; the wrapper chain is shallow-copied and the
        base environment is shared.
        """
        sealed = copy.copy(self)
        sealed.unwrap = False
        if isinstance(sealed.wrapped, Wrapper):
            sealed.wrapped = sealed.wrapped.seal()
        return sealed

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
        if name == "wrapped":
            # If ``wrapped`` itself is missing from __dict__ (e.g. during
            # unpickling or copy, before __init__ has run), delegating the
            # lookup would recurse infinitely.
            raise AttributeError(name)
        return getattr(self.wrapped, name)


class EnvironmentWrapper(Generic[TWrapperState], Wrapper[TWrapperState, Timestep]):
    """Base class for wrappers that only operate on Environments (need reward/terminated/truncated)."""

    wrapped: Environment

    def __init__(self, wrapped: Environment):
        super().__init__(wrapped)
