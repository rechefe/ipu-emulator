# `IPU_EMU_PROTOTYPING`: parse false values case-insensitively

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-04 · Size S · Depends on: none · Behaviour change: yes (`IPU_EMU_PROTOTYPING=FALSE`, `NO`, `Off`, ` 0 ` and similar now mean "off"; today they mean "on")

## Goal

A user who writes `IPU_EMU_PROTOTYPING=FALSE` gets the faithful emulator, in every build.

## Background

Verified at 694a8a8 (`src/tools/ipu-emu-py/src/ipu_emu/prototyping.py`):

- `_FALSEY = {"", "0", "false", "False", "no", "off"}` (prototyping.py:44).
- `default_enabled()` (prototyping.py:47-51) tests `env not in _FALSEY` without normalising case or whitespace. `FALSE`, `NO`, `Off`, `OFF` and `" 0"` all count as a request for prototyping mode.
- `resolve_enabled()` (prototyping.py:59-68) raises `ValueError` when prototyping is requested in a build without `--define=ipu_proto=1`. So in a default build, `IPU_EMU_PROTOTYPING=FALSE` makes every `IpuState()` construction fail with "prototyping mode requires a Bazel build with --define=ipu_proto=1". In a flagged build it silently turns the fast paths on.
- `docs/content/prototyping-mode.md:26` documents the variable as `IPU_EMU_PROTOTYPING=1/0`.

## Decisions applied

- D7: prototyping mode stays. This issue only fixes how its switch is read.

## Implementation

1. In `prototyping.py`:
   ```python
   _FALSEY = frozenset({"", "0", "false", "no", "off"})

   def _env_requests_prototyping(raw: str) -> bool:
       return raw.strip().lower() not in _FALSEY
   ```
   and use it in `default_enabled()`.
2. `prototyping-mode.md:26`: state that the value is case-insensitive and that `0`, `false`, `no`, `off` and the empty string mean off; any other value means on.

## Tests

- [ ] Parametrised unit test of `_env_requests_prototyping` over `FALSE`, `False`, `false`, `NO`, `Off`, `" 0 "`, `""` (all off) and `1`, `true`, `YES`, `on` (all on).
- [ ] Extend `test_bazel_flag_is_a_mandatory_capability` (test_prototyping_equivalence.py:336): in a default build, `IPU_EMU_PROTOTYPING=FALSE` constructs an `IpuState` with `prototyping is False` and does not raise.
- [ ] `bazel test //src/tools/ipu-emu-py/...` and `bazel test --define=ipu_proto=1 //src/tools/ipu-emu-py/...` are green.

## Acceptance Criteria

- [ ] Any capitalisation of a false value, with or without surrounding whitespace, selects the faithful path.
- [ ] The docs table says the value is case-insensitive.
- [ ] The PR is labelled as a behaviour change.

Open question: should an unrecognised value (for example `IPU_EMU_PROTOTYPING=maybe`) raise instead of meaning "on"? This issue keeps today's rule (anything not false is on).
