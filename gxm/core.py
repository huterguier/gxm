import jax
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeVar,
)


TState = TypeVar("TState", bound="State")

@jax.tree_util.register_dataclass
@dataclass
class State:
    time: int


@jax.tree_util.register_dataclass
@dataclass
class EnvState:
    state: State
    obs: jax.Array
    reward: float | jax.Array
    done: bool | jax.Array
    info: dict[str, Any]

    def __getitem__(self, item):
        return (self.state, self.obs, self.reward, self.done, self.info)[item]

    def __iter__(self):
        return iter((self.state, self.obs, self.reward, self.done, self.info))



class Env(Generic[TState]):
    """Base class for environments in gxm."""

    def step(
        self,
        key: jax.Array,
        state: EnvState | TState,
        action: jax.Array,
    ) -> EnvState:
        """Perform a step in the environment given an action."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self, key: jax.Array) -> EnvState:
        """Reset the environment to its initial state."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def num_actions(self) -> int:
        """Return the number of actions available in the environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")



