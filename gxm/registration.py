import importlib

_builtin = {
    "Gymnax": "gxm.adapters.gymnax",
    "Brax": "gxm.adapters.brax",
    "Pgx": "gxm.adapters.pgx",
    "Gymnasium": "gxm.adapters.gymnasium",
    "Craftax": "gxm.adapters.craftax",
"JAXAtari": "gxm.adapters.jaxatari",
    "Navix": "gxm.adapters.navix",
    "XMiniGrid": "gxm.adapters.xminigrid",
}

_registry: dict[str, callable] = {}


def register(library: str, make_fn):
    """
    Register a make function for a new environment library.

    Args:
        library: The library prefix used in environment IDs (e.g. ``"MyLib"``).
        make_fn: A callable ``(id, **kwargs) -> Environment`` that creates an environment
                 given its library-specific ID (without the prefix).
    """
    _registry[library] = make_fn


def make(id: str, **kwargs):
    """
    Create an environment given its ID.

    The ID must be in the format ``"Library/EnvironmentName"``,
    e.g. ``"Gymnax/CartPole-v1"`` or ``"Brax/ant"``.

    Args:
        id: The fully qualified environment ID.
        **kwargs: Additional keyword arguments forwarded to the adapter.
    Returns:
        An instance of the requested environment.
    Raises:
        ValueError: If the library prefix is not recognised.
    """
    library, name = id.split("/", 1)
    if library in _registry:
        return _registry[library](name, **kwargs)
    if library in _builtin:
        module = importlib.import_module(_builtin[library])
        _registry[library] = module.make
        return module.make(name, **kwargs)
    raise ValueError(
        f"Unknown environment library: '{library}'. "
        f"Available built-in libraries: {list(_builtin.keys())}."
    )
