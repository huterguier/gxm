from typing import TYPE_CHECKING

from gxm.core import (
    AutoResetEnvironment,
    Environment,
    EnvironmentState,
    Model,
    ModelState,
    Step,
    Timestep,
    Trajectory,
    Transition,
)
from gxm.registration import make, register

if TYPE_CHECKING:
    from gxm.adapters import (
        brax,
        craftax,
        gymnasium,
        gymnax,
        jaxatari,
        navix,
        pgx,
        xminigrid,
    )

_adapters = frozenset(
    {
        "gymnax",
        "brax",
        "pgx",
        "gymnasium",
        "craftax",
        "jaxatari",
        "navix",
        "xminigrid",
    }
)


def __getattr__(name: str):
    if name in _adapters:
        import importlib

        return importlib.import_module(f"gxm.adapters.{name}")
    raise AttributeError(f"module 'gxm' has no attribute '{name}'")


__all__ = [
    "Model",
    "ModelState",
    "Environment",
    "AutoResetEnvironment",
    "EnvironmentState",
    "Step",
    "Timestep",
    "Transition",
    "Trajectory",
    "make",
    "register",
    "gymnax",
    "brax",
    "pgx",
    "gymnasium",
    "craftax",
    "jaxatari",
    "navix",
    "xminigrid",
]
