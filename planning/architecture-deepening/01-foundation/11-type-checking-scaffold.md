# Type checking as a Bazel test: scaffold now, strict for `ipu_isa` and the reference API first

Part of [Epic E01: Foundation](00-epic.md) · ID E01-11 · Size S · Depends on: E01-02 · Behaviour change: none

## Goal

A Bazel test type-checks a package with the checker version pinned in the lock. The mechanism exists before E03 and E04 create the packages that must be strict, so those issues only add a target with `strict = True` instead of building tooling.

## Background

Measured at 694a8a8 with mypy 2.3:

- No type checker is configured or run anywhere (no `[tool.mypy]`, no `mypy.ini`, no `pyrightconfig.json`, nothing in CI).
- `ipu_common` (14 files): 17 errors in 4 files in default mode, 47 in strict mode. Most are in `isa_alias_spec.py`, which Q31 moves out of the ISA package to `ipu_emu`.
- `ipu_emu` (21 files): about 200 errors in default mode, mostly in the modules E04 rewrites.
- Type hints are already common in new code (`from __future__ import annotations` across `ipu_emu`, typed signatures in `errors.py`, `stats.py`, `prototyping.py`).

## Decisions applied

- Q65: strict typing on `ipu_isa` and the reference-model API first, then spread.
- Q13, Q31: `ipu_isa` is created by E03 (rename of `ipu_common`, minus non-ISA modules). Q11: the reference-model API is a small versioned module from E04 (E04-30).

## Implementation

1. Tool choice: **mypy**. It installs from a pure wheel in the lock and runs inside a Bazel `py_test` with no extra runtime. Pyright's PyPI wrapper downloads Node.js at run time, which breaks the hermetic test. Add `mypy` (and `types-*` stubs only if a strict package needs them) to `tools/requirements-tools.in` and relock.
2. Root `mypy.ini` with shared settings only: `python_version = 3.10`, `warn_unused_configs = True`, `namespace_packages = True`, `explicit_package_bases = True`. Strictness is per target, not global.
3. `tools/typecheck.bzl` with `mypy_test(name, lib, strict = False)`:
   - collects the `srcs` and import roots of `lib` and its first-party deps from `PyInfo`;
   - runs `python -m mypy --config-file mypy.ini [--strict] <lib sources>` with `MYPYPATH` set to those import roots, so imports of other repository packages resolve and are checked only as far as they are used;
   - third-party packages resolve from the same runfiles (numpy ships its own stubs).
4. First target, to prove the mechanism, in default (non-strict) mode with a short baseline:
   ```python
   mypy_test(name = "mypy", lib = ":ipu_common_lib")
   ```
   Fix the 17 default-mode errors in `ipu_common` if they are in files that stay in the ISA package; for `isa_alias_spec.py` (moving out under Q31) add a module-level `# mypy: ignore-errors` with a comment pointing at E03.
5. Write the follow-ups into the issues that own the code, not here:
   - E03 (ISA package): `mypy_test(name = "mypy", lib = ":ipu_isa_lib", strict = True)`.
   - E04-30 (reference-model API): a strict target for the API module.
   - Each later package split adds a default-mode target and a plan to make it strict.
6. Conventions page (E01-12): "new modules have full type hints; packages with a `strict = True` target must stay clean".

## Tests

- [ ] `bazel test //src/tools/ipu-common:mypy` passes.
- [ ] A scratch commit with a deliberate type error (`x: int = "a"`) in `ipu_common` fails it with the file and line.
- [ ] `strict = True` on the same target fails (proves the flag reaches mypy); record the error count in the PR.

## Acceptance Criteria

- [ ] `mypy_test` exists, runs mypy from the lock, and supports `strict`.
- [ ] One target runs in the default tier.
- [ ] E03 and E04-30 each list a strict `mypy_test` in their acceptance criteria (cross-check when they are filed).

Open question: confirm mypy over pyright. Pyright is faster and stricter on inference, but it needs Node.js in the Bazel sandbox (D9 keeps JS out of Bazel).
