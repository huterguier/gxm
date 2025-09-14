<div align="center">
    <img src="https://github.com/huterguier/gxm/blob/main/images/gxm.png" width="200">
</div>

# Unified Functional Interface for RL Environments
[Gxm](https://github.com/huterguier/gxm) aims to be the [Gym](https://www.gymlibrary.dev/)-equivalent for [JAX](https://github.com/jax-ml/jax)-based RL Environments.
It normalizes different environment backends behind one tiny, purely functional API that is `jit`, `vmap` and `scan` friendly and explicit about randomness.
```python
env = gxm.make("Envpool/Breakout-v5")
env_state = env.init(key)
env_state = env.step(env_state, key, action)
```

## Supported Environments
Currently Gxm supports the following Libraries:
- [Gymnax](https://github.com/RobertTLange/gymnax) (Classic Control, bsuite and MinAtar)
- [Pgx](https://github.com/sotetsuk/pgx) (Boardgames and MinAtar)
- [Navix](https://github.com/epignatelli/navix) (Minigrid in JAX)
- [Envpool](https://github.com/sail-sg/envpool) (Vectorized Gymnasium Environements)
- [Craftax](https://github.com/MichaelTMatthews/Craftax) (Crafter in JAX)

## Installation
```
pip install gxm
```
