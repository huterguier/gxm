import jax

from gxm.wrappers import GymnasiumEnv, GymnaxEnv, NavixEnv, PgxEnv


def make(id, **kwargs):
    wrapper, id = id.split("/", 1)
    Wrapper = {
        "Gymnax": GymnaxEnv,
        "Navix": NavixEnv,
        "Gymnasium": GymnasiumEnv,
        "Pgx": PgxEnv,
    }[wrapper]
    return Wrapper(id, **kwargs)


if __name__ == "__main__":

    # env = make("Gymnasium/CartPole-v1")
    env = make("Pgx/minatar-asterix")

    @jax.jit
    def rollout1(key, num_steps=1000):

        def step(state, key):
            key_action, key_step = jax.random.split(key)
            action = jax.random.randint(key_action, (1,), 0, env.num_actions)[0]
            state, obs, reward, done, info = env.step(key_step, state, action)
            return state, None

        state, obs, reward, done, info = env.reset(key)
        keys = jax.random.split(key, num_steps)
        state, _ = jax.lax.scan(step, state, keys)
        return state

    @jax.jit
    def rollout2(key, num_steps=1000):

        def step(env_state, key):
            key_action, key_step = jax.random.split(key)
            action = jax.random.randint(key_action, (1,), 0, env.num_actions)[0]
            env_state = env.step(key_step, env_state, action)
            return env_state, None

        env_state = env.reset(key)
        keys = jax.random.split(key, num_steps)
        env_state, _ = jax.lax.scan(step, env_state, keys)

        return env_state

    print(env.num_actions)
    key = jax.random.PRNGKey(0)
    num_steps = 100
    rollout1(key)
    rollout2(key)
