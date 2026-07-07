from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax

from gxm.core import Environment, EnvironmentState, Model, TStep, Timestep


@jax.tree_util.register_dataclass
@dataclass
class WrapperState(EnvironmentState):
    env_state: EnvironmentState


TWrapperState = TypeVar("TWrapperState", bound=WrapperState)


class Wrapper(Generic[TWrapperState, TStep], Model[TWrapperState, TStep]):
    """Base class for wrappers in gxm, over either a bare Model or an Environment."""

    env: Model[Any, TStep]
    unwrap: bool = True

    def __init__(self, env: Model[Any, TStep], unwrap: bool = True):
        self.env = env
        self.id = env.id
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        if isinstance(env, Wrapper) and not unwrap:
            assert not env.unwrap
        self.unwrap = unwrap

    def has_wrapper(self, wrapper_type: type[Model]) -> bool:
        if isinstance(self, wrapper_type):
            return True
        return self.env.has_wrapper(wrapper_type)

    def get_wrapper(self, wrapper_type: type[Model]) -> Model:
        if isinstance(self, wrapper_type):
            return self
        return self.env.get_wrapper(wrapper_type)

    @property
    def unwrapped(self) -> Model:
        if self.unwrap:
            return self.env.unwrapped
        return self

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(name)


class EnvironmentWrapper(Generic[TWrapperState], Wrapper[TWrapperState, Timestep]):
    """Base class for wrappers that only operate on Environments (need reward/terminated/truncated)."""

    env: Environment

    def __init__(self, env: Environment, unwrap: bool = True):
        super().__init__(env, unwrap=unwrap)
