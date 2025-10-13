<div align="center">
    <img src="https://github.com/huterguier/gxm/blob/main/images/gxm.png" width="170">
</div>

# Unified Functional Interface for RL Environments
[![PyPI version](https://img.shields.io/pypi/v/gxm)](https://pypi.org/project/gxm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-1d8a50.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/codestyle-black-black.svg)](https://opensource.org/licenses/MIT)

[``gxm``](https://github.com/huterguier/gxm) aims to be the [``gym``](https://www.gymlibrary.dev/)-equivalent for [JAX](https://github.com/jax-ml/jax)-based RL Environments.
It normalizes different environment backends behind one tiny, purely functional API that is `jit`, `vmap` and `scan` friendly and explicit about randomness.
For a more detailed description please refer to the [documentation](https://gxm.readthedocs.io/en/latest/).


## Features
- 🤝**Unified Functional Interface:** ``gxm`` unifies different environment libraries behind one tiny API. This eases development and experimentation with different environments.
- 🌐**Broad Environment Support:** ``gxm`` supports a wide range of environments from different libraries. A complete list of supported environments can be found below.
- 💻**CPU based Enironments:** Run your favorite CPU based environments directly in JAX via callbacks. These wrappers also support `vmap` and behave ([almost](https://gxm.readthedocs.io/en/latest/the_sharp_bits.html#cpu-based-environments)) exactly like the other JAX-native environments!
- ✅**Handling Truncation:** ``gxm`` handles truncation and termination in a unified way across all environments. Note that handling trunctation in JAX adds a slight memory overhead but can be disabled if not needed.


## API

Environments in ``gxm`` can be created in the standardized way by using a `make` function.
The identifier strings are of the form `<Library>/<Environment-Name>`.
```python
import gxm
env = gxm.make("Gymnasium/LunarLander-v3")
```
Alternatively you can also be explicit about the environment creation and import the corresponding environment for instantiation.
In both cases environment parameters will be passed to the underlying environment library.
```python
from gxm.environments import GymnasiumEnvironment
env = GymnasiumEnvironment("LunarLander-v3", gravity=-10.0, wind_power=15.0, turbulence_power=1.5)
```
The returned environment object exposes the methods `init`, `step` and `reset`.
Note that there is a clear distinction between `reset` and `init`. 
`init` is used to create a new environment state from scratch while `reset` is used to reset an existing environment state.
For fully functional environments there is no difference between the two, but for CPU based environments `reset` will reuse the existing environment instance while `init` will create a new one.
In addition this conforms to the common JAX pattern of having an `init` function to create an initial state.


```python
env_state, timestep = env.init(key)
for _ in range(1e3):
    env_state, timestep = env.step(key, env_state, action)
env_state, timestep = env.reset(key, env_state)
```
As a reminder, you should never use for loops for environment rollouts in JAX. This is just for demonstration purposes.
A single `timestep` has the following signature in `gxm`. Here `true_obs` stores the observation that would have been observed if there was no truncation.
Hence `obs` and `true_obs` only differ whenever `truncated` is true.
```python
class Timestep:
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    obs: jax.Array
    true_obs: jax.Array
    info: dict[str, Any]
```


## Supported Environments
Currently ``gxm`` supports the following Libraries.
- [Gymnax](https://github.com/RobertTLange/gymnax) (Classic Control, bsuite and MinAtar)
- [Pgx](https://github.com/sotetsuk/pgx) (Boardgames and MinAtar)
- [Navix](https://github.com/epignatelli/navix) (Minigrid in JAX)
- [Envpool](https://github.com/sail-sg/envpool) (Vectorized Gymnasium Environements)
- [Craftax](https://github.com/MichaelTMatthews/Craftax) (Crafter in JAX)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (Classic Control, Atari, Box2D, MuJoCo, etc.)

The following environments are planned to be supported in the future.
- [Brax](https://github.com/google/brax) (Physics-based Environments in JAX)
- [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) (Physics-based Environments in Python)
- [Jumanji](https://github.com/instadeepai/jumanji) (Various RL Environments in JAX)


## Installation
``gxm`` can be installed directly from PyPI.
```
pip install gxm
```
By default Gxm comes without any of the underlying environment libraries.
You can install any combination of them by using optional dependencies or all of the at once using ``all``.
```
pip install gxm[gymnax, pgx, navix, envpool, craftax, gymnasium]
```

## Citation
If you use ``gxm`` in your research, please cite it as follows.
Please also cite the underlying environment libraries that you used. Their Githubs are linked above.
```bibtex
@software{gxm2025github,
  author = {Henrik Metternich},
  title = {{gxm}: Unified Functional Interface for RL Environments in JAX},
  url = {https://github.com/huterguier/gxm},
  version = {0.1.1},
  year = {2025},
}
```
