# ipu-common

Shared types and definitions for all IPU Python packages.

## Overview

`ipu-common` provides the single source of truth for:

- **Register descriptors** (`RegDescriptor`): Metadata for all IPU registers
- **Data types** (`RegDtype`): Register storage formats (uint8, int8, uint32, int32, uint128)
- **Register kinds** (`RegKind`): Pipeline stages (mult, acc, lr, cr, misc)

## Usage

```python
from ipu_common import RegDtype, RegKind, RegDescriptor

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
bazel build //src/tools/ipu-common:ipu_common_lib
```

## Updating

After modifying `pyproject.toml`, regenerate Bazel dependencies:

```bash
bazel run //src/tools/ipu-common:update_requirements
```
