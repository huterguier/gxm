import jax

from gxm.wrappers import GymnaxEnv, NavixEnv


def make(id, **kwargs):
    wrapper, id = id.split("/", 1)
    Wrapper = {
        "Gymnax": GymnaxEnv,
        "Navix": NavixEnv,
    }[wrapper]
    return Wrapper(id, **kwargs)


if __name__ == "__main__":
    import gymnax

    env = make("Navix/Navix-FourRooms-v0")

    @jax.jit
    def rollout1(key, num_steps=100000):

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
    def rollout2(key, num_steps=100):
        env_state = env.reset(key)
        for _ in range(num_steps):
            action = jax.random.randint(key, (1,), 0, env.num_actions)[0]
            env_state = env.step(key, env_state, action)
            _, obs, reward, done, info = env_state
            print(
                f"Step: {env_state[0].time}, Action: {action}, Reward: {reward}, Done: {done}"
            )

    print(env.num_actions)
    key = jax.random.PRNGKey(0)
    num_steps = 100
    rollout1(key)
