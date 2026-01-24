# Quick Start

## 1. Installation

By default, `gxm` comes without any environment backends. You can install support for specific backends by specifying them in the extras.
Below is an example of installing support for both `pgx` and `gymnasium` backends.

```bash
pip install 'gxm[pgx,gymnasium]'
```

## 2. Creating an Environment

Use the `gxm.make` function to create an environment. The environment ID must be specified in the format `"Library/EnvironmentName"`.
```python
import gxm
env = gxm.make("Envpool/Breakout-v5")
```
Below are some example environment IDs for different backends.

| Library Backend | Example ID                             |
|-----------------|----------------------------------------|
| Gymnax          | Gymnax/CartPole-v1                     |
| Envpool         | Envpool/Breakout-v5                    |
| Gymnasium       | Gymnasium/MountainCarContinuous-v0     |
| Craftax         | Craftax/Craftax-Classic-v1             |

## 3. Spaces

Each `gxm` environment has `observation_space` and `action_space` attributes that define the spaces for observations and actions, respectively.
In the following example, we use these spaces to correctly initialize a neural network with the appropriate input and output dimensions.
Note that the property `n` is only defined for discrete spaces and composite spaces containing discrete spaces only.

```python
import jax
import flax.linen as nn

class Network(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x

network = Network(env.action_space.n)
params = network.init(key, jax.numpy.zeros(env.observation_space.shape))
```


## 4. Basic Environment Lifecycle

Interaction with the environment is done through the functional methods `init`, `step`, and `reset`, all of which require a JAX random key. 
The environment state is can be initialized using `init`.
Note that there is a clear distinction between `reset` and `init`.
`init` only requires a random key as input and returns a new environment state and the initial timestep.
`reset`, on the other hand, requires both a random key and the current environment state as input, and it returns a new environment state and the initial timestep for a new episode.
The output of these functions is always a tuple of `(EnvironmentState, Timestep)`. The `EnvironmentState` must be passed into the next `step` or `reset` call, as it contains the environment's current internal data.


```python
key = jax.random.PRNGKey(0)
key, key_init, key_step, key_reset = jax.random.split(key)

env_state, timestep = env.init(key_init)

action = env.action_space.sample(key_action)
env_state, timestep = env.step(key_step, env_state, action)

reset_env_state, reset_timestep = env.reset(key_reset, next_env_state)
```
