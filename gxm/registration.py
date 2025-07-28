import jax
from gxm.wrappers import GymnaxEnv


def make(env_id, **kwargs):
    return GymnaxEnv(env_id, **kwargs)


if __name__ == "__main__":

    def rollout1(env, key, num_steps):
        state, obs, reward, done, info = env.reset(key)
        for _ in range(num_steps):
            action = jax.random.randint(key, (1,), 0, env.num_actions)[0]
            state, obs, reward, done, info = env.step(key, state, action)
            print(f"Step: {state.time}, Action: {action}, Reward: {reward}, Done: {done}")
            if done:
                break

    def rollout2(env, key, num_steps):
        env_state = env.reset(key)
        for _ in range(num_steps):
            action = jax.random.randint(key, (1,), 0, env.num_actions)[0]
            env_state = env.step(key, env_state, action)
            _, obs, reward, done, info = env_state
            print(f"Step: {env_state[0].time}, Action: {action}, Reward: {reward}, Done: {done}")
            if done:
                break

    env = make("CartPole-v1")
    print(env.num_actions)
    key = jax.random.PRNGKey(0)
    num_steps = 100
    rollout1(env, key, num_steps)
    rollout2(env, key, num_steps)
