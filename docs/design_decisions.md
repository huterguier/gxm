# Design Decisions

This page outlines the key design decisions made during the development of the `gxm` API.


##
- needs termination and truncation
- different auto reset models
- only one way is really compatible with jax
- termination and truncation require a second observation
-> the api would get really bloated
-> in the gymnasium style api we need an additional obs and state variable making it 7 in total
```python
state, obs, reward, done, info = env.step(key, state, action)
```
```python
state, obs, true_obs, reward, terminated, truncated, info = env.step(key, state, action)
```
- we decided to instead bundle everything into a single object
```python
env_state, timestep = env.step(key, env_state, action)
```



