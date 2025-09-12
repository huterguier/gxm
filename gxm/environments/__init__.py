try:
    from gxm.environments.craftax_environment import CraftaxEnvironment
except ImportError:
    CraftaxEnvironment = None
try:
    from gxm.environments.gymnax_environment import GymnaxEnvironment
except ImportError:
    GymnaxEnvironment = None
try:
    from gxm.environments.pgx_environment import PgxEnvironment
except ImportError:
    PgxEnvironment = None
try:
    from .envpool_environment import EnvpoolEnvironment
except (ImportError, AttributeError):
    if AttributeError:
        print("Envpool is not compatible with JAX>=0.5.0 by default. Check the installation guide.")
    EnvpoolEnvironment = None

__all__ = [
    "GymnaxEnvironment",
    "PgxEnvironment",
    "CraftaxEnvironment",
    "EnvpoolEnvironment",
]
