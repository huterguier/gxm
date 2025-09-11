# Gxm <small><em>(Gym for JAX)</em></small>

[Gxm](https://github.com/huterguier/gxm) aims to be the [Gym](https://www.gymlibrary.dev/)-equivalent for [JAX](https://github.com/jax-ml/jax)-based RL Environments.
It normalizes different environment backends behind one tiny, (purely) functional API that is `jit`, `vmap` and `scan` friendly and explicit about randomness.
```python
env = gxm.make("Envpool/Breakout-v5")
env_state = env.init(key)
env_state = env.step(env_state, key, action)
```

## Installation

```bash
pip install gxm
```
By default only the `gymnax` backend is installed. To install additional backends use one or more of the following extras.
```bash
pip install gxm[envpool,pgx,gymnasium,craftax,mjx,brax]
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Introduction

quick_start
the_sharp_bits
examples
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: API

design_decisions
api
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Environments

environments/gymnax
environments/envpool
environments/pgx
environments/gymnasium
craftax
mjx
brax
```
