import jax
import jax.numpy as jnp
import pytest

from gxm.core import Environment


def assert_matching_structure(a, b, what: str):
    """Assert two pytrees have identical structure, leaf shapes and dtypes."""
    assert jax.tree.structure(a) == jax.tree.structure(b), (
        f"{what}: pytree structures differ:\n"
        f"  {jax.tree.structure(a)}\n  {jax.tree.structure(b)}"
    )
    for path_a, leaf_a in jax.tree_util.tree_flatten_with_path(a)[0]:
        leaf_b = {p: leaf for p, leaf in jax.tree_util.tree_flatten_with_path(b)[0]}[
            path_a
        ]
        leaf_a, leaf_b = jnp.asarray(leaf_a), jnp.asarray(leaf_b)
        assert leaf_a.shape == leaf_b.shape, (
            f"{what}: leaf {jax.tree_util.keystr(path_a)} shape differs: "
            f"{leaf_a.shape} vs {leaf_b.shape}"
        )
        assert leaf_a.dtype == leaf_b.dtype, (
            f"{what}: leaf {jax.tree_util.keystr(path_a)} dtype differs: "
            f"{leaf_a.dtype} vs {leaf_b.dtype}"
        )


def trees_equal(a, b) -> bool:
    return all(
        jax.tree.leaves(jax.tree.map(lambda x, y: jnp.array_equal(x, y), a, b))
    )


class TestEnvironment:
    __test__ = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__test__ = True

    @pytest.fixture(params=[])
    def env(request) -> Environment:
        raise NotImplementedError("Add environments to the fixture parameters.")

    def test_init(self, env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        assert env_state is not None
        assert timestep is not None

    def test_reset(self, env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        env_state, timestep = env.reset(key, env_state)
        assert env_state is not None
        assert timestep is not None

    def test_step(self, env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        action = env.action_space.sample(key)
        env_state, timestep = env.step(key, env_state, action)
        assert env_state is not None
        assert timestep is not None

    def test_vmap_init(self, env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        assert env_states is not None
        assert timesteps is not None

    def test_vmap_reset(self, env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        env_states, timesteps = jax.vmap(env.reset)(keys, env_states)
        assert env_states is not None
        assert timesteps is not None

    def test_vmap_step(self, env: Environment):
        key = jax.random.key(0)
        keys = jax.random.split(key, 10)
        env_states, timesteps = jax.vmap(env.init)(keys)
        actions = env.action_space.sample(key, (10,))
        env_states, timesteps = jax.vmap(env.step)(keys, env_states, actions)
        assert env_states is not None
        assert timesteps is not None

    # -- Conformance: the semantic contract every environment must satisfy. --

    def test_init_and_step_return_same_structure(self, env: Environment):
        """init and step outputs must be interchangeable as scan carries:
        identical pytree structure, leaf shapes, and dtypes."""
        key = jax.random.key(0)
        env_state_init, timestep_init = env.init(key)
        action = env.action_space.sample(key)
        env_state_step, timestep_step = env.step(key, env_state_init, action)
        assert_matching_structure(env_state_init, env_state_step, "env_state")
        assert_matching_structure(timestep_init, timestep_step, "timestep")

    def test_scan_carry(self, env: Environment):
        """(env_state, timestep) must survive as a jax.lax.scan carry."""
        key = jax.random.key(0)
        carry = env.init(key)

        def step_fn(carry, key):
            env_state, _ = carry
            action = env.action_space.sample(key)
            env_state, timestep = env.step(key, env_state, action)
            return (env_state, timestep), timestep.reward

        jax.lax.scan(step_fn, carry, jax.random.split(key, 8))

    def test_init_timestep_is_episode_start_sentinel(self, env: Environment):
        """The init timestep marks an episode start: done is True (its next_obs
        begins a fresh episode) and true_next_obs equals next_obs. Its action
        and reward are placeholders and must never enter a replay buffer."""
        key = jax.random.key(0)
        _, timestep = env.init(key)
        assert bool(timestep.done), "init timestep must have done=True"
        assert trees_equal(timestep.next_obs, timestep.true_next_obs), (
            "init timestep must have true_next_obs == next_obs"
        )

    def test_true_next_obs_equals_next_obs_unless_done(self, env: Environment):
        """On non-terminal steps the two observation fields must be identical;
        they may only diverge when done (auto-reset replaced next_obs)."""
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        for i in range(32):
            key, key_step = jax.random.split(key)
            action = env.action_space.sample(key_step)
            env_state, timestep = env.step(key_step, env_state, action)
            if bool(timestep.done):
                env_state, timestep = env.reset(key_step, env_state)
            else:
                assert trees_equal(timestep.next_obs, timestep.true_next_obs), (
                    f"true_next_obs diverged from next_obs on non-done step {i}"
                )

    def test_observation_in_space(self, env: Environment):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        assert bool(env.observation_space.contains(timestep.next_obs)), (
            "init observation not contained in observation_space"
        )
        for _ in range(8):
            key, key_step = jax.random.split(key)
            action = env.action_space.sample(key_step)
            env_state, timestep = env.step(key_step, env_state, action)
            assert bool(env.observation_space.contains(timestep.next_obs)), (
                "step observation not contained in observation_space"
            )
