from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tqdx
from jax import Array

import gxm
from gxm.wrappers import ClipReward, EpisodicLife, RecordEpisodeStatistics


@jax.tree_util.register_dataclass
@dataclass
class PQNState:
    key: Array
    params: Any
    opt_state: Any
    env_state: Any
    timestep: Any
    info: Any


class PQN:

    def __init__(self, args, env, network, optimizer):
        self.args = args
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.args["n_loops"] = int(
            self.args["n_steps"] // (self.args["n_steps_rollout"] * self.args["n_envs"])
        )

    def init(self, key):
        key_init, key_network = jax.random.split(key, 2)
        keys_init = jax.random.split(key_init, self.args["n_envs"])
        env_state, timestep = jax.vmap(self.env.init)(keys_init)
        params = self.network.init(
            key_network, jax.tree.map(lambda x: x[0], timestep.obs)
        )
        opt_state = self.optimizer.init(params)
        info = {"step": 0, "episode": 0, "update": 0}

        return PQNState(
            key=key,
            info=info,
            params=params,
            opt_state=opt_state,
            env_state=env_state,
            timestep=timestep,
        )

    def rollout(self, key, alg_state, n_steps):

        def pi_epsilon(key, alg_state, obs):
            key_e, key_action = jax.random.split(key)
            epsilon = optax.linear_schedule(
                self.args["e_start"],
                self.args["e_end"],
                self.args["e_fraction"] * self.args["n_steps"],
            )(alg_state.info["step"])
            action_random = self.env.action_space.sample(key_action)
            q = self.network.apply(alg_state.params, obs)
            action_greedy = jnp.argmax(q, axis=-1)
            action = jnp.where(
                jax.random.uniform(key_e) < epsilon, action_random, action_greedy
            )
            return action, q

        def step(alg_state, key):
            obs = alg_state.timestep.obs
            keys_pi, keys_step = jax.random.split(key, (2, self.args["n_envs"]))
            action, q = jax.vmap(pi_epsilon, (0, None, 0))(keys_pi, alg_state, obs)
            env_state, timestep = jax.vmap(self.env.step)(
                keys_step, alg_state.env_state, action
            )
            transition = timestep.transition(prev_obs=obs, action=action)
            alg_state.env_state = env_state
            alg_state.timestep = timestep
            alg_state.info["step"] += self.args["n_envs"]

            return alg_state, (transition, q)

        keys = jax.random.split(key, n_steps)
        alg_state, (transitions, qs) = jax.lax.scan(step, alg_state, keys)
        return alg_state, transitions, qs

    def minibatches(self, key, alg_state, transitions, qs, n_minibatches):
        def lambda_returns(timesteps, qs, last_q):
            def lambda_step(carry, x):
                g_next, q_next = carry
                reward, qs, done = x
                delta = g_next - q_next
                g = reward + self.args["gamma"] * (1 - done) * (
                    (1 - self.args["lambda"]) * q_next + self.args["lambda"] * g_next
                )
                q = jnp.max(qs, axis=-1)
                return (g, q), g + delta

            g_t = (
                timesteps.reward[-1]
                + self.args["gamma"] * (1 - timesteps.done[-1]) * last_q
            )
            q_t = jnp.max(qs[-1], axis=-1)
            _, returns = jax.lax.scan(
                lambda_step,
                (g_t, q_t),
                (timesteps.reward, qs, timesteps.done),
                reverse=True,
            )
            return returns

        last_q = jax.vmap(self.network.apply, (None, 0))(
            alg_state.params, transitions.obs[-1]
        )
        targets = lambda_returns(transitions, qs, jnp.max(last_q, axis=-1))
        batches = (transitions, targets)
        batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batches)
        batch = jax.tree.map(lambda x: jax.random.permutation(key, x), batch)
        minibatches = jax.tree.map(
            lambda x: x.reshape(n_minibatches, -1, *x.shape[1:]),
            batch,
        )
        return minibatches

    def update(self, alg_state, minibatches, n_update_epochs):

        def q_pred(params, sample):
            qs_pred = self.network.apply(params, sample.prev_obs)
            q_pred = qs_pred[sample.action]
            return q_pred

        def loss(params, minibatch):
            batch, targets = minibatch
            qs_pred = jax.vmap(q_pred, (None, 0))(params, batch)
            loss = jnp.mean((qs_pred - targets) ** 2)
            return loss

        def update_step(state, minibatch):
            grads = jax.grad(loss)(state.params, minibatch)
            updates, state.opt_state = self.optimizer.update(grads, state.opt_state)
            state.params = optax.apply_updates(state.params, updates)
            state.info["update"] += 1
            return state, None

        def update_epoch(state_minibatches, _):
            state, minibatches = state_minibatches
            state = jax.lax.scan(update_step, state, minibatches)[0]
            state_minibatches = (state, minibatches)
            return state_minibatches, None

        (alg_state, _), _ = jax.lax.scan(
            update_epoch, (alg_state, minibatches), length=n_update_epochs
        )

        return alg_state

    def loop(self, key, alg_state):
        alg_state, samples, qs = self.rollout(
            key, alg_state, self.args["n_steps_rollout"]
        )
        minibatches = self.minibatches(
            key, alg_state, samples, qs, self.args["n_minibatches"]
        )
        alg_state = self.update(alg_state, minibatches, self.args["n_update_epochs"])
        jax.debug.print(
            "Step: {s}, Update: {u}, Return: {r}",
            s=alg_state.info["step"],
            u=alg_state.info["update"],
            r=alg_state.timestep.info["episodic_return"].mean(),
        )
        return alg_state

    def train(self, key, init_alg_state=None):

        def loop(alg_state, key):
            alg_state = self.loop(key, alg_state)
            return alg_state, None

        key_init, key_eval, key_epoch = key, *jax.random.split(key)
        if init_alg_state is None:
            alg_state = self.init(key_init)
        else:
            alg_state = init_alg_state

        alg_state, _ = tqdx.scan(
            loop,
            alg_state,
            jax.random.split(key_epoch, self.args["n_loops"]),
        )

        return alg_state


class Network(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (1, 2, 0))
        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = x.reshape(-1)
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x.squeeze()


if __name__ == "__main__":
    args = {
        "n_steps": 5e7,
        "n_envs": 16,
        "n_steps_rollout": 256,
        "n_minibatches": 32,
        "n_update_epochs": 2,
        "gamma": 0.995,
        "lambda": 0.65,
        "lr": 2.5e-4,
        "max_grad_norm": 10.0,
        "e_start": 1.0,
        "e_end": 0.0005,
        "e_fraction": 0.1,
    }
    import ale_py

    env = gxm.make("Gymnasium/ALE/Breakout-v5", reward_clipping=False)
    env = RecordEpisodeStatistics(env)
    env = EpisodicLife(env)
    env = ClipReward(env)
    network = Network(action_dim=env.action_space.n)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args["max_grad_norm"]), optax.radam(args["lr"])
    )

    alg = PQN(args, env, network, optimizer)
    agent_state = jax.jit(alg.train)(jax.random.key(0))
