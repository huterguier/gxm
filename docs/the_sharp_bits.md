---
title: 🔪The Sharp Bits🔪
---
# 🔪The Sharp Bits🔪

## CPU-based Environments
CPU-based environemnts like `envpool` and `gymnasium` are not actually functional.
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

