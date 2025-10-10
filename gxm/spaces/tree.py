from typing import Any, Tuple

import jax
import jax.numpy as jnp

from .space import Space

Shape = Tuple[int, ...]


class Tree(Space):
    """Space composed of multiple subspaces in a tree structure."""

    def __init__(self, spaces: Any):
        self.spaces = spaces

    def sample(self, key: jax.Array, shape: Shape = ()) -> Any:
        """
        Sample random action from all subspaces, retaining the original structure.

        Args:
            key: JAX random key.
            shape: Shape of the sample to be drawn from each subspace.
        Returns:
            A tree-structured object with samples from each subspace.
        """
        structure = jax.tree.structure(
            self.spaces, is_leaf=lambda x: isinstance(x, Space)
        )
        keys = jax.random.split(key, structure.num_leaves)
        key_tree = jax.tree.unflatten(structure, keys)
        return jax.tree.map(
            lambda space, k: space.sample(k, shape), self.spaces, key_tree
        )

    def contains(self, x: jax.Array) -> bool:
        """
        Check whether dimensions of object are within subspace.

        Args:
            x: Object to be checked.
        Returns:
            True if object is within all subspaces, False otherwise.
        """
        structure = jax.tree.structure(
            self.spaces, is_leaf=lambda x: isinstance(x, Space)
        )
        x_structure = jax.tree.structure(
            x, is_leaf=lambda y: isinstance(y, (jnp.ndarray, int, float))
        )
        if structure != x_structure:
            return False
        return jax.tree.reduce(
            lambda a, b: a and b,
            jax.tree.map(lambda space, y: space.contains(y), self.spaces, x),
            True,
        )

    @property
    def n(self) -> int:
        return jax.tree.reduce(
            lambda a, b: a * b.n,
            self.spaces,
            1,
            lambda x: isinstance(x, Space),
        )

    def __repr__(self) -> str:
        return f"Tree({self.spaces})"
