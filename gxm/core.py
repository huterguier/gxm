from dataclasses import dataclass
from typing import Any

import jax


@jax.tree_util.register_dataclass
@dataclass
class EnvironmentState:
    state: Any
    obs: jax.Array
    reward: float | jax.Array
    done: bool | jax.Array
    info: dict[str, Any]

    def __getitem__(self, item):
        return (self.state, self.obs, self.reward, self.done, self.info)[item]

    def __iter__(self):
        return iter((self.state, self.obs, self.reward, self.done, self.info))


class Environment:
    """Base class for environments in gxm."""

    def init(self, key: jax.Array) -> EnvironmentState:
        """Initialize the environment and return the initial state."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def step(
        self,
        key: jax.Array,
        env_state: EnvironmentState,
        action: jax.Array,
    ) -> EnvironmentState:
        """Perform a step in the environment given an action."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def reset(self, key: jax.Array, env_state: EnvironmentState) -> EnvironmentState:
        """Reset the environment to its initial state."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def num_actions(self) -> int:
        """Return the number of actions available in the environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")
