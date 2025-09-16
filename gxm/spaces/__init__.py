"""Spaces for defining action and observation spaces."""

from gxm.spaces.box import Box
from gxm.spaces.discrete import Discrete
from gxm.spaces.space import Space
from gxm.spaces.tree import Tree

__all__ = [
    "Discrete",
    "Box",
    "Tree",
    "Space",
]
