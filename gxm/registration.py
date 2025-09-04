import jax

from gxm.wrappers import GymnaxEnvironment, PgxEnvironment, EnvpoolEnvironment


def make(id, **kwargs):
    wrapper, id = id.split("/", 1)
    Wrapper = {
        "Gymnax": GymnaxEnvironment,
        "Pgx": PgxEnvironment,
        "Envpool": EnvpoolEnvironment,
    }[wrapper]
    return Wrapper(id, **kwargs)


if __name__ == "__main__":

    # env = make("Gymnax/CartPole-v1")
    env = make("Envpool/Pong-v5")

    @jax.jit
    def rollout(key, num_steps=1000):

        def step(env_state, key):
            key_action, key_step = jax.random.split(key)
            action = jax.random.randint(key_action, (1,), 0, env.num_actions)[0]
            env_state = env.step(key_step, env_state, action)
            jax.debug.print("{}", env_state.done)
            return env_state, None

        env_state = env.init(key)
        keys = jax.random.split(key, num_steps)
        env_state, _ = jax.lax.scan(step, env_state, keys)

        return env_state

    key = jax.random.PRNGKey(0)
    num_steps = 100
    env_state = rollout(key)
