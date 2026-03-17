from gxm.environments import (
    BraxEnvironment,
    CraftaxEnvironment,
    EnvpoolEnvironment,
    GymnasiumEnvironment,
    GymnaxEnvironment,
    JAXAtariEnvironment,
    NavixEnvironment,
    PgxEnvironment,
    XMiniGridEnvironment,
)

_registry = {
    "Gymnax": GymnaxEnvironment,
    "Pgx": PgxEnvironment,
    "Envpool": EnvpoolEnvironment,
    "Craftax": CraftaxEnvironment,
    "XMiniGrid": XMiniGridEnvironment,
    "JAXAtari": JAXAtariEnvironment,
    "Gymnasium": GymnasiumEnvironment,
    "Navix": NavixEnvironment,
    "Brax": BraxEnvironment,
}


def register(library: str, environment_class):
    """
    Register a new environment library.

    Args:
        library (str): The name of the library to register.
        environment_class: The environment class to associate with the library.
    """
    _registry[library] = environment_class


def make(id: str, **kwargs):
    """
    Create an environment given its id.
    The id should be in the format "Library/EnvironmentName", e.g. "Gymnax/CartPole-v1".

    Args:
        id (str): The id of the environment to create.
        **kwargs: Additional keyword arguments to pass to the environment constructor.
    Returns:
        An instance of the requested environment.
    Raises:
        ValueError: If the library is not recognized.

    Examples:
        >>> env = make("Gymnax/CartPole-v1")
        >>> env = make("Pgx/MountainCarContinuous-v0")
        >>> env = make("Envpool/Pong-v5")
    """
    library = id.split("/", 1)[0]
    if library not in _registry:
        raise ValueError(f"Unknown environment library: {library}")
    Environment = _registry[library]
    return Environment(id, **kwargs)

