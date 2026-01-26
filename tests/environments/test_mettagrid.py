import jax
import pytest
from test_environment import TestEnvironment

import gxm


class TestMettagrid(TestEnvironment):

    @pytest.fixture(params=[6, 12])
    def num_agents(self, request):
        # num_agents must be divisible by 6 (agents per instance in default arena)
        return request.param

    @pytest.fixture
    def env(self, num_agents):
        return gxm.make("Mettagrid/arena", num_agents=num_agents)

    def test_init(self, env):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        assert env_state is not None
        assert timestep is not None
        assert timestep.obs.shape[0] == env.num_agents
        assert timestep.reward.shape[0] == env.num_agents

    def test_reset(self, env):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        env_state, timestep = env.reset(key, env_state)
        assert env_state is not None
        assert timestep is not None
        assert timestep.obs.shape[0] == env.num_agents

    def test_step(self, env):
        key = jax.random.key(0)
        env_state, timestep = env.init(key)
        # Multi-agent: sample actions for all agents
        actions = env.action_space.sample(key, (env.num_agents,))
        env_state, timestep = env.step(key, env_state, actions)
        assert env_state is not None
        assert timestep is not None
        assert timestep.obs.shape[0] == env.num_agents
        assert timestep.reward.shape[0] == env.num_agents

    def test_vmap_init(self, env):
        key = jax.random.key(0)
        keys = jax.random.split(key, 4)
        env_states, timesteps = jax.vmap(env.init)(keys)
        assert env_states is not None
        assert timesteps is not None
        assert timesteps.obs.shape[0] == 4
        assert timesteps.obs.shape[1] == env.num_agents

    def test_vmap_reset(self, env):
        key = jax.random.key(0)
        keys = jax.random.split(key, 4)
        env_states, timesteps = jax.vmap(env.init)(keys)
        env_states, timesteps = jax.vmap(env.reset)(keys, env_states)
        assert env_states is not None
        assert timesteps is not None

    def test_vmap_step(self, env):
        key = jax.random.key(0)
        keys = jax.random.split(key, 4)
        env_states, timesteps = jax.vmap(env.init)(keys)
        # Multi-agent: sample actions for all agents across all environments
        actions = env.action_space.sample(key, (4, env.num_agents))
        env_states, timesteps = jax.vmap(env.step)(keys, env_states, actions)
        assert env_states is not None
        assert timesteps is not None

    def test_rollout(self, env):
        """Test a multi-step rollout with random actions."""
        key = jax.random.key(0)
        env_state, timestep = env.init(key)

        total_reward = 0
        for i in range(10):
            key, subkey = jax.random.split(key)
            actions = env.action_space.sample(subkey, (env.num_agents,))
            env_state, timestep = env.step(subkey, env_state, actions)
            total_reward += timestep.reward.sum()

        assert env_state is not None
        assert timestep is not None

    def test_spaces(self, env):
        """Test that action and observation spaces are properly set."""
        from gxm.spaces import Box, Discrete

        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Box)
        assert env.num_agents > 0
