# Error hierarchy under `IpuError`; asserts become exceptions

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-03 · Size S · Depends on: E03 (`IpuError` root in `ipu_isa`) · Behaviour change: none for programs (invalid API calls raise a typed exception instead of `AssertionError`)

## Goal

Give the emulator two error kinds (Q59, Q66) on one root, and make every check survive `python -O` (L1):

- `EmulatorFault` — a hardware fault F-1 to F-9. E04-19 fills it in; this issue only creates the class.
- `ConfigurationError` — bad setup (format, program size, prototyping without the build flag, invalid API arguments). Raised at setup time only, never mid-run.

## Background

- `errors.py` defines one class, `EmulatorError(RuntimeError)`.
- Other raises are untyped: `RuntimeError` for the LR conflict (ipu.py:1203, 2244) and the cycle limit (emulator.py:76, 122); `ValueError` for XMEM bounds (xmem.py:60, 73), program size (emulator.py:48) and the prototyping build gate (prototyping.py:65); `IndexError` for an undefined opcode (plan tables in `_build_slot_plans`, ipu.py:502).
- `regfile.py` has 13 `assert` statements (scalar index, register byte length, generated accessors). `ipu.py` has 7 (`assert self._snapshot_buffer is not None`, `Partition` type checks at 846, 868). All disappear under `python -O`.

## Implementation

1. In `ipu_emu/errors.py`:
   ```python
   from ipu_isa.errors import IpuError

   class EmulatorError(IpuError): ...            # root of everything ipu_emu raises
   class EmulatorFault(EmulatorError): ...       # hardware fault; fields added in E04-19
   class ConfigurationError(EmulatorError, ValueError): ...
   ```
   `EmulatorError` stops subclassing `RuntimeError`. A grep of `src/` at HEAD finds no `except RuntimeError`; tests that use `pytest.raises(RuntimeError)` for an `EmulatorError` are updated in the same PR. Re-run the grep before merging.
2. Replace the 13 regfile asserts and the 7 ipu.py asserts with two helpers, `_check_index(name, idx, limit)` (raises `IndexError`) and `_check_len(name, data, size)` (raises `ConfigurationError`). The snapshot `assert`s become a single invariant in the cycle path (`Cycle` owns the snapshot after E04-07; until then raise `EmulatorError`).
3. Raise `ConfigurationError` from the existing setup-time checks: program too large (emulator.py:48), prototyping gate (prototyping.py:65), `AliasProfile` reuse (ipu.py:583).
4. Leave the mid-run raises (`RuntimeError` LR conflict, XMEM `ValueError`, `IndexError` opcode, cycle limit) untouched. E04-19 maps them to fault IDs as a behaviour change.

## Tests

- [ ] New Bazel test variant `test_regfile_optimized` runs `test_regfile.py` with `env = {"PYTHONOPTIMIZE": "1"}` and asserts the typed exceptions.
- [ ] `ConfigurationError` is an `IpuError`, an `EmulatorError` and a `ValueError` (so existing `pytest.raises(ValueError)` tests stay green).
- [ ] All emulator and app tests pass unchanged, which proves no program-visible change. E02 kernel output digests unchanged.

## Acceptance Criteria

- [ ] `grep -n "assert " src/ipu_emu/regfile.py src/ipu_emu/ipu.py` returns nothing.
- [ ] Every class in `ipu_emu.errors` derives from `ipu_isa.errors.IpuError`.
- [ ] Setup-time errors raise `ConfigurationError`.
