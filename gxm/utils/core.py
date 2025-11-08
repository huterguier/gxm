import jax
import jax.numpy as jnp

from gxm.core import Environment, Trajectory
from gxm.typing import Key, Policy, PolicyState


def rollout(
    key: Key,
    env: Environment,
    pi: Policy,
    pi_state: PolicyState,
    n_steps: int,
) -> Trajectory:
    """
    Perform a rollout in the environment using the given policy.

    Args:
        key: A JAX random key for any stochasticity in the environment and policy.
        env: The environment to perform the rollout in.
        pi: The policy to use for selecting actions.
        pi_state: The initial state of the policy.
        n_steps: The number of steps to perform in the rollout.
    Returns:
        A Trajectory object containing the observations, actions, rewards, and other data from the rollout.
    """

    def step(carry, _):
        key, env_state, pi_state, timestep = carry
        key, key_pi, key_step = jax.random.split(key, 3)
        action, pi_state = pi(key_pi, pi_state, timestep.observation)
        env_state, timestep = env.step(key_step, env_state, action)
        carry = (key, env_state, pi_state, timestep)
        return carry, (timestep, action)

    env_state, timestep = env.init(key)
    carry = (key, env_state, pi_state, timestep)
    carry, (timesteps, actions) = jax.lax.scan(step, carry, None, length=n_steps)
    traj = timesteps.trajectory(timestep.obs, actions)
    return traj


def evaluate(
    key: Key,
    env: Environment,
    pi: Policy,
    pi_state: PolicyState,
    n_steps: int,
):
    """
    Evaluate a policy in the environment by performing a rollout and computing the mean return.

    Args:
        key: A JAX random key for any stochasticity in the environment and policy.
        env: The environment to perform the evaluation in.
        pi: The policy to evaluate.
        pi_state: The initial state of the policy.
        n_steps: The number of steps to perform in the rollout.
    Returns:
        The mean return of the policy over the rollout.
    """
    traj = rollout(key, env, pi, pi_state, n_steps=n_steps)

    def step(carry, x):
        r = carry
        reward, done = x
        r = reward + (1.0 - done) * r
        return r, r

    rs, _ = jax.lax.scan(step, 0.0, (traj.reward, traj.done))
    mean_return = jnp.sum(traj.done * rs) / jnp.sum(traj.done)
    return mean_return
