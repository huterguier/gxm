# Design Decisions

This page documents the key design decisions behind the `gxm` API.
The goal is to explain not just *what* the API looks like, but *why* it was designed that way,
including the tradeoffs that were considered and rejected.

---

## The `Timestep` Object

### Why bundle the step output into a single object?

Most RL libraries follow the Gymnasium convention of returning multiple values from `step`:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

This works well for simple environments, but breaks down once truncation is handled correctly.
Proper truncation handling requires exposing a second observation — the true next state before
the auto-reset kicks in — so the agent can bootstrap correctly at episode boundaries.
In the flat Gymnasium style, this would look like:

```python
env_state, obs, true_obs, reward, terminated, truncated, info = env.step(key, env_state, action)
```

That is seven return values, three of which are observations or observation-like. This is unwieldy,
and gets worse as wrappers start adding more fields. Every wrapper would need to thread all seven
values through its interface, and callers would need to unpack them every time.

`gxm` bundles all of this into a single `Timestep` object:

```python
env_state, timestep = env.step(key, env_state, action)
```

**Pros:**
- The function signature stays clean regardless of how many fields are added.
- Wrappers only need to accept and return a `Timestep`, not a growing tuple.
- Fields are accessed by name, not by position — no risk of confusing `obs` with `true_obs`.
- The object can carry additional fields (e.g. `info`) without changing any function signatures.

**Cons:**
- Slightly more verbose to access individual fields: `timestep.reward` vs. just `reward`.
- Users coming from Gymnasium need to adjust to a new pattern.

The verbosity cost is small and one-time. The clarity and extensibility benefits are permanent.

---

## Naming: `next_obs` over `obs`

### What does `Timestep.next_obs` represent?

A `Timestep` represents the result of taking an action at step $i$:
the reward $R_i$ and the resulting observation $S_{i+1}$.
The observation field therefore always refers to the observation *after* the action was taken —
it is, by definition, the *next* observation.

### Why not just call it `obs`?

`obs` is the dominant convention in RL libraries. Gymnasium returns `obs` from `step`,
even though it is technically the next observation. This convention is widely understood
and has the advantage of brevity.

However, `gxm` uses `next_obs` for a specific reason: `Transition` inherits from `Timestep`.

### The `Transition` inheritance argument

A `Transition` represents the full tuple $(S_t, A_t, R_t, S_{t+1})$ and inherits all fields
from `Timestep` — including the observation field. It then adds `prev_obs` to represent $S_t$.

If `Timestep` used `obs` for the next observation, `Transition` would have:
- `prev_obs` — the observation before the action
- `obs` — the observation after the action (inherited)

This asymmetry is confusing. The two fields represent the same kind of data at different
timesteps, but their names give no indication of that relationship.

With `next_obs`, `Transition` has:
- `prev_obs` — the observation before the action
- `next_obs` — the observation after the action (inherited)

This pair is symmetric, self-documenting, and matches the naming convention used by most
RL replay buffer implementations.

**Summary of options considered:**

| Option | `Timestep` field | `Transition` fields | Verdict |
|--------|-----------------|---------------------|---------|
| Short, conventional | `obs` | `prev_obs` / `obs` | Asymmetric in `Transition` |
| Explicit, consistent | `next_obs` | `prev_obs` / `next_obs` | Chosen |
| Overly verbose | `observation` | `prev_observation` / `next_observation` | Unnecessary verbosity |

**Pros of `next_obs`:**
- Semantically accurate — the field always holds the observation that follows the action.
- Symmetric with `prev_obs` in `Transition`, making both fields immediately readable together.
- Removes any ambiguity in the `Transition` context where two observations coexist.

**Cons of `next_obs`:**
- Less familiar to users coming from Gymnasium, which uses `obs` for the same thing.
- Slightly more verbose, especially for `true_next_obs`.

---

## The `true_next_obs` field and truncation

### What is truncation and why does it require a second observation?

In RL, an episode can end in two ways:

- **Termination**: the agent reaches a true terminal state (e.g. falling over, winning the game).
  The value of the next state is zero — no bootstrapping is needed.
- **Truncation**: the episode is cut short by an external constraint (e.g. a time limit).
  The environment is *not* in a terminal state; the agent simply ran out of steps.
  Bootstrapping from the next state is still needed for correct value estimation.

When auto-reset is active (as in most training loops), the environment immediately resets after
truncation, so the observation returned after the last step is the *first* observation of the
*next* episode — not the observation of the state the agent was truncated in.

If an agent uses this reset observation for bootstrapping, it will bootstrap from a completely
unrelated state, which corrupts the value estimate and destabilises training.

### How `gxm` handles it

`Timestep` exposes two observation fields:

- `next_obs`: the observation returned by the environment after the step, post-reset if truncated.
- `true_next_obs`: the observation at the state the agent was actually in when truncation occurred.

These two fields are identical in all cases *except* when `truncated` is `True`.
When writing a training loop, use `true_next_obs` for bootstrapping:

```python
# Correct bootstrapping under truncation
bootstrap_obs = jnp.where(timestep.truncated, timestep.true_next_obs, timestep.next_obs)
value_target = timestep.reward + gamma * (1 - timestep.terminated) * critic(bootstrap_obs)
```

If you do not need truncation handling (e.g. your environments never truncate, or you are
using `IgnoreTruncation`), `true_next_obs` will equal `next_obs` and can be ignored.

**Pros:**
- Correct value estimation at episode boundaries without any special casing in the environment.
- The distinction is explicit in the API — it is impossible to accidentally use the wrong observation.

**Cons:**
- Doubles the memory footprint of the observation for every truncated step.
- `true_next_obs` is a somewhat verbose name. The `IgnoreTruncation` wrapper sets it to `None`
  if the field is not needed.

---

## `Transition` as a flat subclass of `Timestep`

### Why does `Transition` inherit from `Timestep`?

The alternative to inheritance is composition: a `Transition` could hold two `Timestep` objects
plus an action.

```python
# Nested composition (rejected)
class Transition:
    prev_timestep: Timestep
    next_timestep: Timestep
    action: PyTree
```

This looks clean, but has a fundamental semantic mismatch. A `Timestep` in `gxm` represents
$(R_i, S_{i+1})$ — the *result* of taking an action. It bundles a reward with the *subsequent*
observation. There is no `Timestep` that cleanly represents "the state before the action" because
that state has no associated reward yet.

Concretely, `prev_timestep.reward` in a nested `Transition` would refer to the reward from the
step *before* the current transition — which is not part of this transition at all. This is
actively misleading.

The flat inheritance model avoids this entirely. `Transition` simply extends `Timestep` with the
fields that were not yet available at step time (`prev_obs`, `action`, `prev_info`):

```python
class Transition(Timestep):
    prev_obs: PyTree
    prev_info: dict[str, PyTree]
    action: PyTree
```

All fields live at the same level, access is ergonomic, and the semantics are correct.

**Pros of flat inheritance:**
- No nested attribute access (`transition.reward` not `transition.next_timestep.reward`).
- No semantic mismatch — every field in `Transition` belongs to this transition.
- Works naturally with `jax.vmap` and `jax.lax.scan`, which expect flat pytrees.
- Converting a `Timestep` to a `Transition` is a natural extension, not a restructuring.

**Cons of flat inheritance:**
- Dataclass inheritance requires care with field ordering and `register_dataclass`.
- The relationship between `Timestep` and `Transition` is not immediately obvious to new users.
