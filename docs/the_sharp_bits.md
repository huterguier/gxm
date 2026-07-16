---
title: 🔪The Sharp Bits🔪
---
# 🔪The Sharp Bits🔪


## CPU-based Environments
CPU-based environments like `gymnasium` are not actually functional under the hood.
Hence, they can be used for sequential rollouts only.
Trying to call the step function from the same state twice will result in an error or unexpected behavior.

```python
import gxm, jax

env = gxm.make("Gymnasium/CartPole-v1")
key = jax.random.key(0)
env_state = env.init(key)
next_env_state = env.step(env_state, key, 0)
# This will cause an error or unexpected behavior
next_env_state = env.step(env_state, key, 0)
```

Usually this is not an issue, as environments are typically used in a sequential manner.
However specialized algorithms, such as [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search), are not compatible with these environments.

## ``init`` vs ``reset``
Typically, RL environments have a `reset` function that resets the environment to an initial state.
In ``gxm``, there is a clear distinction between `init` and `reset`. 
- `init`: This function initializes the environment and returns the initial state.
- `reset`: This function resets the environment to a new initial state after an episode ends. It can be called multiple times during the lifecycle of an environment.

This distinction is motivated by two key aspects:
1. **Consistency**: Most JAX libraries use the ``init`` function to initialize states. 
   By using `init` for the initial state, we maintain consistency with other JAX libraries.
2. **Compatibility**: CPU-based environments like gymnasium maintain a reference to the environment instance on the host.
   Calling ``reset`` without this state would lead to an instantiation of a new environment, which is not the intended behavior.

## Brax on CPU
Brax's physics pipelines are no longer actively maintained (brax itself recommends MJX),
and repeatedly creating and compiling brax environments in a single long-lived CPU process
has been observed to intermittently crash inside jaxlib's XLA compiler
(`Fatal Python error: Segmentation fault` in `backend_compile_and_load`). The crash is
nondeterministic and unrelated to gxm's adapter; re-running typically succeeds. Prefer
reusing environment instances over re-creating them, and prefer GPU jaxlib where available.

## JAXAtari
JAXAtari's ``PixelObsWrapper`` auto-resets internally on the same step an episode
ends, with no way to opt out. The pre-reset terminal observation is destroyed
inside jaxatari, so — unlike every other adapter — ``true_next_obs`` always equals
``next_obs`` for JAXAtari environments, and bootstrapping across truncated episode
ends is *not* corrected. In addition, jaxatari requires sprite assets before any
environment can be created: run ``install-sprites`` once (declining the ROM
ownership prompt installs the freely distributable replacement sprites).

## Craftax pixel observation spaces
Craftax's pixel environments declare a transposed `(W, H, C)` observation space while
actually emitting `(H, W, C)` observations. The gxm adapter corrects the declared space
from a probe observation at construction time.

## Navix observation bounds
Some navix environments declare observation bounds that their observations exceed
(e.g. `Navix-Empty-5x5-v0` declares a maximum entity id of 8 but emits ids up to 10).
When a probe observation escapes the declared space, the gxm adapter widens the bound
to the observation dtype's full range at construction time.

## Termination and Truncation
Handling termination and truncation correctly is essential for the correctness of many reinforcement learning algorithms.
Yet most JAX-based environment libraries do not account for this at all.
Handling truncation is not problematic when handling resets manually, but becomes tricky when using auto-resets, as the true observation is lost when the environment is reset.
While on CPU-based environments the true observation can be returned conditionally, in JAX this is not possible as the returned structure must be known at compile time.
This means that the only way to handle truncation correctly is to always return the true observation, even if there was no truncation at that step.

In the example below, the `IgnoreTruncation` wrapper will treat truncation as termination, setting the `terminated` flag to `True` when truncation occurs. In addition it sets `true_next_obs` equal to `next_obs` for all timesteps, since the truncation/termination distinction is no longer meaningful once truncation is folded into termination.

```python
import gxm
from gxm.wrappers import IgnoreTruncation
env = IgnoreTruncation(gxm.make("Gymnasium/ALE/Breakout-v5"))
```

## Composite Spaces
In most environment libraries, different classes are used to represent composite spaces such as dictionaries, tuples, and other structured combinations. 
`gxm` unifies composite spaces by relying on PyTrees, eliminating the need for separate classes. 
This allows for arbitrary compositions, such as tuples of dictionaries of tuples, and simplifies extension to new types of spaces, as long as they follow the pytree structure.
```python
from gxm.spaces import Discrete, Tree
space = Tree({
    "position": Discrete(5),
    "velocity": Discrete(3),
    "info": {
        "goal": Discrete(2),
        "obstacles": (Discrete(4), Discrete(4))
    }
})
```

## Vmapping over functions that contain `init`

Let's say you have written a small rollout function that takes in a single key and an action and returns the environment state after 10 steps with that action.

```python
env = gxm.make("Gymnasium/LunarLander-v3")
def rollout(key, action):
    def step(env_state, _):
        env_state, _ = env.step(key, env_state, action)
        return env_state, None

    env_state, _ = env.init(key)
    env_state, _ = jax.lax.scan(step, env_state, None, length=10)
    return env_state
```

You can vmap over the `rollout` function to perform multiple rollouts in parallel.
Assuming that you don't care much about randomness you might be tempted to do something like this.

```python
key = jax.random.key(0)
actions = env.action_space.sample(key, (5))
batched_rollout = jax.vmap(rollout, (None, 0))(key, actions)
```

For all JAX-native environments this will work as expected since JAX can infer that the shape of the environment. However for all CPU-based environments this will fail as the `init` function looks at the key to determine the number of environments to create.
To fix this you need to make sure that you're vmapping over the keys as well.

```python
keys = jax.random.split(key, 5)
batched_rollout = jax.vmap(rollout)(keys, actions)
```
