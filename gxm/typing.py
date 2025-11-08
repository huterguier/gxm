from typing import Any, Callable, TypeAlias

import jax

Array: TypeAlias = jax.Array
Key: TypeAlias = jax.Array
PyTree: TypeAlias = Any

Action: TypeAlias = PyTree
Observation: TypeAlias = PyTree

PolicyState: TypeAlias = PyTree
Policy: TypeAlias = Callable[[Key, PolicyState, Observation], Action]

QState: TypeAlias = PyTree
Q: TypeAlias = Callable[[Key, QState, Observation, Action], Array]
