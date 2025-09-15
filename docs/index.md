# Gxm <small><em>(Gym for JAX)</em></small>

[Gxm](https://github.com/huterguier/gxm) aims to be the [Gym](https://www.gymlibrary.dev/)-equivalent for [JAX](https://github.com/jax-ml/jax)-based RL Environments.
It normalizes different environment backends behind one tiny, (purely) functional API that is `jit`, `vmap` and `scan` friendly and explicit about randomness.
```python
import gxm

env = gxm.make("Envpool/Breakout-v5")
env_state, timestep = env.init(key)
env_state, timestep = env.step(env_state, key, action)
env_state, timestep = gxm.reset(key, env_state)
```

## Environments
Currently Gxm supports the following Libraries:
- [Gymnax](https://github.com/RobertTLange/gymnax) (Classic Control, bsuite and MinAtar)
- [Pgx](https://github.com/sotetsuk/pgx) (Boardgames and MinAtar)
- [Navix](https://github.com/epignatelli/navix) (Minigrid in JAX)
- [Envpool](https://github.com/sail-sg/envpool) (Vectorized Gymnasium Environements)
- [Craftax](https://github.com/MichaelTMatthews/Craftax) (Crafter in JAX)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (Classic Control, Atari, Box2D, MuJoCo, etc.)

The following environments are planned to be supported in the future:
- [Brax](https://github.com/google/brax) (Physics-based Environments in JAX)
- [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) (Physics-based Environments in Python)
- [Jumanji](https://github.com/instadeepai/jumanji) (Various RL Environments in JAX)

## Installation

```bash
pip install gxm
```
By default only the `gymnax` backend is installed. To install additional backends use one or more of the following extras.
```bash
pip install gxm[envpool,pgx,gymnasium,craftax,mjx,brax]
```

## Citation
If you use Gxm in your research, please cite the repository and the underlying environment libraries.
```txt
@misc{metternich2025gxm,
      title={Gxm: Unified Functional Interface for RL Environments in JAX},
      author={Henrik Metternich},
      year={2024},
      publisher={GitHub},
      journal={GitHub repository},
      howpublished={\url{https://arxiv.org/abs/2406.08352}},
}
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
api/gxm
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
