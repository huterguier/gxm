from gxm.environments import (
    CraftaxEnvironment,
    EnvpoolEnvironment,
    GymnasiumEnvironment,
    GymnaxEnvironment,
    JAXAtariEnvironment,
    NavixEnvironment,
    PgxEnvironment,
    XMiniGridEnvironment,
)


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
    Environment = {
        "Gymnax": GymnaxEnvironment,
        "Pgx": PgxEnvironment,
        "Envpool": EnvpoolEnvironment,
        "Craftax": CraftaxEnvironment,
        "XMiniGrid": XMiniGridEnvironment,
        "JAXAtari": JAXAtariEnvironment,
        "Gymnasium": GymnasiumEnvironment,
        "Navix": NavixEnvironment,
    }[library]
    return Environment(id, **kwargs)
