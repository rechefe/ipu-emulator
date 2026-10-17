# RegFile: replace the validation `assert`s with real exceptions

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-02 · Size S · Depends on: none · Behaviour change: yes (a bad register index or a wrong-length write raises `RegisterIndexError` / `RegisterSizeError` instead of `AssertionError`, and still raises under `python -O`)

## Goal

Every bounds and length check in `regfile.py` holds under `python -O`. A negative index or a wrong-length buffer can no longer write to the wrong place or resize a register.

## Background

Verified at 694a8a8 (`src/tools/ipu-emu-py/src/ipu_emu/regfile.py`). The register file validates its inputs with 13 `assert` statements:

| Line | Accessor | Check |
|---|---|---|
| 151, 152 | `get_scalar` | scalar register; `0 <= index < count` |
| 167, 168 | `set_scalar` | scalar register; `0 <= index < count` |
| 199 | `get_register_bytes` | index range |
| 208, 209 | `set_register_bytes` | index range; `len(data) == size_bytes` |
| 392 | generated `set_<name>` (single blob, e.g. `set_r_mask`) | `len(data) == size` |
| 412 | generated `get_<name>` (indexed, e.g. `get_r`) | index range |
| 421, 422 | generated `set_<name>` (indexed, e.g. `set_r`) | index range; `len(data) == elem_size` |
| 517 | generated `get_<name>_word` | word index range |
| 525 | generated `set_<name>_word` | word index range |

Under `python -O` all of them disappear. Then a negative index reads or writes from the end of the backing `bytearray` (`struct.unpack_from` / `pack_into` and slicing accept negative offsets), and a wrong-length slice assignment (`self._storage[n][start:end] = data`) silently grows or shrinks the register's storage.

These checks are reachable from a program, not only from Python callers. `test_cached_scalar_read_invalid_index_exception_is_bit_exact` (test_prototyping_equivalence.py:291-312) patches an `lr_inst` immediate field to CR index 999, and the cycle raises `AssertionError` from `get_scalar`.

Existing tests pin the assertion type: `test_regfile.py:65, 98, 103, 171` and `test_prototyping_equivalence.py:306`.

`errors.py` holds one class, `EmulatorError(RuntimeError)`.

## Decisions applied

- L1 (accepted via Q29): asserts used as validation become exceptions.
- Q38, Q59: the final error model (hardware faults F-1 to F-9 versus configuration errors) comes with E03/E04. E04-03 re-parents these classes under `IpuError`, and E04-19 turns the program-reachable case (an out-of-range register index decoded from an instruction word) into a fault. This issue only uses temporary types that already behave like the built-ins, so callers that catch `IndexError` / `ValueError` or `EmulatorError` keep working when E04-03 moves them.

## Implementation

1. In `ipu_emu/errors.py` add:
   ```python
   class RegisterIndexError(EmulatorError, IndexError):
       """Register or word index outside the register's declared count."""

   class RegisterSizeError(EmulatorError, ValueError):
       """Data written to a register does not match the register's byte size."""

   class RegisterKindError(EmulatorError, TypeError):
       """Scalar accessor used on a vector register."""
   ```
   (`class X(RuntimeError, IndexError)` is a valid layout; checked on CPython 3.10.)
2. In `regfile.py` add two module-level helpers and use them at all 13 sites:
   ```python
   def _check_index(name: str, index: int, count: int) -> None:
       if not 0 <= index < count:
           raise RegisterIndexError(f"{name}[{index}] out of range (count={count})")

   def _check_size(name: str, data: bytes | bytearray, size: int) -> None:
       if len(data) != size:
           raise RegisterSizeError(f"{name}: expected {size} bytes, got {len(data)}")
   ```
   The `is_vector` checks in `get_scalar` / `set_scalar` raise `RegisterKindError`. Keep the existing messages so log output stays the same.
3. Export the three classes from `ipu_emu/__init__.py` next to `EmulatorError`.
4. Do not touch the `assert self._snapshot_buffer is not None` and `Partition` type asserts in `ipu.py`. They are internal invariants and belong to E04-03.

## Tests

- [ ] `test_regfile.py`: the four `pytest.raises(AssertionError)` become the specific class. Add negative-index cases (`set_lr(-1, 0)`, `get_r(-1)`, `set_r_acc_word(-1, 0)`) and a wrong-length case for each generated setter kind (single blob, indexed, word).
- [ ] New test: run a register-index check in a subprocess under `sys.executable -O` and assert it still raises `RegisterIndexError`, so the `-O` guarantee is tested, not assumed.
- [ ] `test_prototyping_equivalence.py:306` expects `RegisterIndexError` in both modes, and the existing args-equality check still passes.
- [ ] `bazel test //...` and `bazel test --define=ipu_proto=1 //src/tools/ipu-emu-py/...` are green.

## Acceptance Criteria

- [ ] `grep -n "assert " src/tools/ipu-emu-py/src/ipu_emu/regfile.py` returns nothing.
- [ ] No valid program changes behaviour. Only the exception type on invalid input changes.
- [ ] The PR is labelled as a behaviour change and lists the new exception classes.

## Risks

- `src/` code that catches `AssertionError` from the register file. A grep at HEAD finds none in `src/tools/*/src` (the `pytest.raises(AssertionError)` in `attention/test.py:417, 610` wrap `np.testing.assert_allclose`, not the register file). Re-run the grep before merging.
