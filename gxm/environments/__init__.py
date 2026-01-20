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
    from gxm.environments.gymnasium_environment import GymnasiumEnvironment
except ImportError:
    GymnasiumEnvironment = None
try:
    from .envpool_environment import EnvpoolEnvironment
except:
    EnvpoolEnvironment = None
try:
    from gxm.environments.jaxatari_environmnet import JAXAtariEnvironment
except ImportError:
    JAXAtariEnvironment = None
try:
    from gxm.environments.xminigrid_environment import XMiniGridEnvironment
except ImportError:
    XMiniGridEnvironment = None
try:
    from gxm.environments.navix_environmnent import NavixEnvironment
except ImportError:
    NavixEnvironment = None
try:
    from gxm.environments.brax_environment import BraxEnvironment
except ImportError:
    BraxEnvironment = None

__all__ = [
    "GymnaxEnvironment",
    "PgxEnvironment",
    "CraftaxEnvironment",
    "EnvpoolEnvironment",
    "JAXAtariEnvironment",
    "GymnasiumEnvironment",
    "XMiniGridEnvironment",
    "NavixEnvironment",
    "BraxEnvironment",
]