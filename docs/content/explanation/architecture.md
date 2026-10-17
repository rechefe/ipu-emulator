# Architecture

This page will explain the packages, their one-way dependencies, and how the emulator executes one instruction word per cycle.

[E09-17](https://github.com/rechefe/ipu-emulator/blob/master/planning/architecture-deepening/09-docs/17-explanation-architecture.md) fills this page.

## Stable and internal code

Only the reference model API is stable: the names of `ipu_emu.reference`, the
commit records its `step()` returns and the commit record and trace schemas.
The [Reference model API](../reference/reference-model-api.md) page lists
them and says what a breaking change is. Everything else in `ipu_emu`, and
every other package's Python API, is internal and can change in any release.
