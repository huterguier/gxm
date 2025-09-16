<div align="center">
    <img src="https://github.com/huterguier/gxm/blob/main/images/gxm.png" width="200">
</div>

# Unified Functional Interface for RL Environments
[Gxm](https://github.com/huterguier/gxm) aims to be the [Gym](https://www.gymlibrary.dev/)-equivalent for [JAX](https://github.com/jax-ml/jax)-based RL Environments.
It normalizes different environment backends behind one tiny, purely functional API that is `jit`, `vmap` and `scan` friendly and explicit about randomness.
```python
env = gxm.make("Envpool/Breakout-v5")
env_state, timestep = env.init(key)
env_state, timestep = env.step(key, env_state, action)
env_state, timestep = env.reset(key, env_state)
```

## Supported Environments
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
```
pip install gxm
```
