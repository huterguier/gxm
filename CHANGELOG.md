# Changelog

All notable changes to this project are documented in this file. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); pre-1.0, so the minor
version carries breaking changes.

## [Unreleased]

## [0.4.0] - 2026-07-09

### Added
- `Dynamics`/`DynamicsState` (formerly `Model`/`ModelState`): reward-free base interface
  for world models.
- `EnvironmentWrapper`: base for wrappers needing `reward`/`terminated`/`truncated`.
- `StickyAction`, `Discretize`, `FlattenObservation`, `StepCounter` can now wrap a bare
  `Dynamics`, not just an `Environment`.

### Changed
- **Breaking:** `Model` → `Dynamics`, `ModelState` → `DynamicsState`.
- **Breaking:** `Wrapper.env` (attribute + constructor param) → `Wrapper.wrapped`.
- **Breaking:** `WrapperState.env_state` → `WrapperState.wrapped_state`.
- **Breaking:** `reset`/`step`'s state parameter is now uniformly named `state` (was
  `env_state`), fixing an incompatible-override bug.
- Formatting/linting: `black`+`isort` → `Ruff`.

### Fixed
- `docs/the_sharp_bits.md`: corrected `IgnoreTruncation`'s `true_next_obs` behavior
  (set to `next_obs`, not `None`) and dropped a dead `Rollout` reference.
