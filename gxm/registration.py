import jax

from gxm.environments import (
    CraftaxEnvironment,
    EnvpoolEnvironment,
    GymnaxEnvironment,
    PgxEnvironment,
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
    library, id = id.split("/", 1)
    Environment = {
        "Gymnax": GymnaxEnvironment,
        "Pgx": PgxEnvironment,
        "Envpool": EnvpoolEnvironment,
        "Craftax": CraftaxEnvironment,
    }[library]
    return Environment(id, **kwargs)


if __name__ == "__main__":

    env = make("Gymnax/CartPole-v1")
    # env = make("Envpool/Breakout-v5")

    @jax.jit
    def rollout(key, num_steps=1000):

        def step(env_state, key):
            key_action, key_step = jax.random.split(key)
            action = jax.random.randint(key_action, (1,), 0, env.num_actions)[0]
            env_state, timestep = env.step(key_step, env_state, action)
            jax.debug.print("{}", timestep.done)
            return env_state, None

        env_state, _ = env.init(key)
        keys = jax.random.split(key, num_steps)
        env_state, _ = jax.lax.scan(step, env_state, keys)

        return env_state

    key = jax.random.PRNGKey(0)
    num_steps = 100
    env_state = rollout(key)
