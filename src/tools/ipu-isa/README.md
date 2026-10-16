# ipu-isa

The IPU instruction set, as data: the one package every other IPU package
depends on. It holds ISA definitions only; the activation math and the MAC-alias
catalogue live in `ipu_emu`, and generators live in `ipu_gen`.

## Overview

`ipu-isa` provides the single source of truth for:

- **Instructions** (`instruction_spec.py`): every slot, instruction and operand
- **Element formats** (`formats.py`): `ElementFormat` (`int8`, `e1`..`e7`, and the
  emulator-only `fp32`, `int32`) with the element bytes, row bytes, accumulator
  type and register element layout derived from each
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
bazel build //src/tools/ipu-isa:ipu_isa_lib
```

## Updating

Python dependencies are declared in `pyproject.toml` and pinned, with hashes, in
the repository-root `requirements_lock.txt`. Runtime dependencies are also listed
as `RUNTIME_DEPS` in the package's `deps.bzl`, which its `py_library` uses. After
changing a dependency, refresh the lock (`//tools:deps_consistency_test` checks
all three agree):

```bash
bazel run //tools:update_requirements_lock
```
