import jax

from gxm import Environment


def _test_environmnet(env: Environment):
    def test_init(env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        assert env_state is not None
        assert timestep is not None

    def test_reset(env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        env_state, timestep = env.reset(key, env_state)
        assert env_state is not None
        assert timestep is not None

    def test_step(env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        action = env.action_space.sample(key)
        env_state, timestep = env.step(key, env_state, action)
        assert env_state is not None
        assert timestep is not None

    def test_vmap_init(env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        assert env_states is not None
        assert timesteps is not None

    def test_vmap_reset(env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        env_states, timesteps = jax.vmap(env.reset)(keys, env_states)
        assert env_states is not None
        assert timesteps is not None

    def test_vmap_step(env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        actions = env.action_space.sample(key, (10,))
        env_states, timesteps = jax.vmap(env.step)(keys, env_states, actions)
        assert env_states is not None
        assert timesteps is not None

    test_init(env)
    test_reset(env)
    test_step(env)
    test_vmap_init(env)
    test_vmap_reset(env)
    test_vmap_step(env)
