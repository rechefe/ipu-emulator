# ipu-isa

The IPU instruction set, as data: the one package every other IPU package
depends on. It holds ISA definitions only; the activation math and the MAC-alias
catalogue live in `ipu_emu`, and generators live in `ipu_gen`.

## Overview

`ipu-isa` provides the single source of truth for:

- **Instructions** (`instruction_spec.py`): every slot, instruction and operand,
  authored as dicts and read only by the loader
- **The typed view** (`view.py`): `isa()` returns the frozen `Isa` (instructions,
  pseudo-instructions, slots, operand types, layout); every consumer reads it
- **Instruction word** (`codec/word.py`): `WORD_LAYOUT` (every field, its wire name
  and bit position), `DecodedWord`, `word_bytes`, `hardware_word`, and the program
  file layout (`pack_program`, `unpack_program`)
- **Run modes** (`formats.py`): `ElementFormat` (`nx8`, the hardware mode, and
  the emulator-only `fp32`, ADR 0007) with the element bytes, row bytes,
  accumulator type and register element layout derived from each
- **NX8** (`nx8.py`): the quantizer and expansion of the AaQ and MULT specs
  (`encode_row`, `decode_element`, the dynamic-exponent rule) and the row
  metadata word
- **CR contents** (`cr_halves.py`): the QF and DS half layouts, the MULT
  constant layout and the half-CR operand codes
- **AaQ LUT** (`lut.py`): the LUT's bit layout, its five segments and
  `lut_image`, which builds the rows that load a function (entry format
  provisional, OQ-47)
- **Timing contract** (`timing.py`): `SLOT_EXECUTION_ORDER`, the cycle-start /
  forwarded rules (`contract_view`), and `KNOWN_DEVIATIONS`, the reads where the
  emulator differs from the contract today; each instruction declares its
  register reads and writes in `instruction_spec.py`
- **Faults** (`faults.py`): the hardware faults F-1 to F-9 (`FaultId`, `FAULTS`:
  priority, detector, modelled, enabled at reset), the `ArithmeticEvent` values,
  and `reserved_encodings`, F-1's reserved opcodes and operand codes derived from
  the ISA
- **Errors** (`errors.py`): `IpuError`, the root of every IPU tooling error
- **Register descriptors** (`RegDescriptor`): Metadata for all IPU registers
- **Data types** (`RegDtype`): Register storage formats (uint8, int8, uint32, int32, uint128)
- **Register kinds** (`RegKind`): Pipeline stages (mult, acc, lr, cr, misc)

## Usage

```python
from ipu_isa import RegDtype, RegKind, RegDescriptor

# Define a register
r_acc_desc = RegDescriptor(
    name="r_acc",
    kind=RegKind.ACC,
    size_bytes=512,
    dtype=RegDtype.UINT8,
    word_view=True,
)
```

## Building

```bash
bazel build //src/ipu-isa:ipu_isa_lib
```

## Updating

Python dependencies are declared in `pyproject.toml` and pinned, with hashes, in
the repository-root `requirements_lock.txt`. Runtime dependencies are also listed
as `RUNTIME_DEPS` in the package's `deps.bzl`, which its `py_library` uses. After
changing a dependency, refresh the lock (`//tools:test_deps_consistency` checks
all three agree):

```bash
bazel run //tools:update_requirements_lock
```
