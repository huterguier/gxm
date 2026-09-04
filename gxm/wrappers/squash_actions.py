import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np

from gxm.core import Dynamics, DynamicsState, TStep
from gxm.spaces import Box
from gxm.typing import Key, PyTree
from gxm.wrappers.wrapper import Wrapper


class SquashActions(Wrapper[Any, TStep]):
    r"""
    Wrapper that accepts unbounded actions and squashes them into the wrapped
    action space with :math:`\tanh`.

    The wrapped dynamics receive :math:`\tanh(a)`, rescaled from :math:`[-1, 1]`
    to the bounds of the wrapped :class:`~gxm.spaces.Box`. The returned step keeps
    the *unsquashed* action :math:`a`, so that an algorithm learning from the step
    sees exactly what its policy produced. This lets a Gaussian policy act in an
    unbounded space and evaluate its log-probabilities there, avoiding the
    numerically unstable inverse of :math:`\tanh` at the bounds.

    >>> import gxm
    >>> from gxm.wrappers import SquashActions
    >>> env = SquashActions(gxm.make("Gymnasium/Pendulum-v1"))

    The action space is left as the wrapped one, so its shape is unchanged; any
    real-valued action of that shape is accepted.
    """

    wrapped: Dynamics[Any, TStep]

    def __init__(self, wrapped: Dynamics[Any, TStep]):
        """
        Args:
            wrapped: The dynamics to wrap. Its action space must be a
                :class:`~gxm.spaces.Box` with finite bounds.
        """
        super().__init__(wrapped)
        space = wrapped.action_space
        if not isinstance(space, Box):
            raise TypeError(
                f"SquashActions requires a Box action space, got {type(space).__name__}"
            )
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("SquashActions requires finite action bounds")
        self.center = jnp.asarray((high + low) / 2.0)
        self.half_range = jnp.asarray((high - low) / 2.0)

    def squash(self, action: PyTree) -> PyTree:
        return self.center + self.half_range * jnp.tanh(action)

    def init(self, key: Key) -> tuple[DynamicsState, TStep]:
        return self.wrapped.init(key)

    def reset(self, key: Key, state: DynamicsState) -> tuple[DynamicsState, TStep]:
        return self.wrapped.reset(key, state)

    def step(
        self,
        key: Key,
        state: DynamicsState,
        action: PyTree,
    ) -> tuple[DynamicsState, TStep]:
        state, step = self.wrapped.step(key, state, self.squash(action))
        return state, dataclasses.replace(step, action=action)
