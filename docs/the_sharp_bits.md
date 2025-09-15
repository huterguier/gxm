---
title: 🔪The Sharp Bits🔪
---
# 🔪The Sharp Bits🔪


## CPU-based Environments
CPU-based environemnts like `envpool` and `gymnasium` are not actually functional under the hood.
Hence, they can be used for sequential rollouts only.
Trying to call the step function from the same state twice will result in an error or unexpected behavior.

```python
import gxm, jax

env = gxm.make("Envpool/CartPole-v1")
key = jax.random.key(0)
env_state = env.init(key)
next_env_state = env.step(env_state, key, 0)
# This will cause an error or unexpected behavior
next_env_state = env.step(env_state, key, 0)
```

Usually this is not an issue, as environments are typically used in a sequential manner.
However specialized algorithms, such as MCTS, are not compatible with these environments.

## ``init`` vs ``reset``
Typically, RL environments have a `reset` function that resets the environment to an initial state.
In ``gxm``, there is a clear distinction between `init` and `reset`. 
- `init`: This function initializes the environment and returns the initial state. It is called once at the beginning of an episode.
- `reset`: This function resets the environment to a new initial state after an episode ends. It can be called multiple times during the lifecycle of an environment.

This distinvtion is motivated by two reasons:
1. **Consistency**: Most JAX libraries use the ``init`` function to initialize states. 
   By using `init` for the initial state, we maintain consistency with other JAX libraries.
2. **Compatibility**: CPU-based environments like `envpool` and gymnasium maintain a state.
   Calling ``reset`` without this state would lead to an instantiation of a new environment, which is not the intended behavior.

## Using ``envpool``
Envpool is no longer maintained. The non-optional XLA-interface relies on JAX<0.4.27
and importing envpool unavoidably leads to a an error due to breaking changes in JAX.
In order to use ``envpool``, you need to uncomment overwrite the following lines in ``envpool/python/xla_interface.py``:
```python
# import jaxlib.xla_extension as xla_extension
# from jaxlib import xla_client as xla_extension
```

