# Building IPU Applications

This guide shows how to build a complete IPU application using the fully connected neural network layer as an example. The complete code is in [src/tools/ipu-apps/src/ipu_apps/fully_connected](https://github.com/rechefe/ipu-emulator/tree/master/src/tools/ipu-apps/src/ipu_apps/fully_connected).

## Application Structure

Each IPU application is a subpackage under `ipu_apps/` containing:

1. **Assembly code** (`.asm`) — IPU program with compute operations (see [Assembly Syntax Guide](assembly-syntax.md))
2. **Python app class** (`__init__.py`) — Subclass of `IpuApp` that implements `setup()` and `teardown()`
3. **Debug runner** (`__main__.py`) — Optional CLI entry point for interactive debugging
4. **Test data** — Input/output binary files for validation
5. **Regression tests** (`test/test_*.py`) — Automated tests comparing outputs against golden references

Everything lives together in one directory.

## Step 1: Write the Assembly Program

Create your IPU assembly program (e.g., `fully_connected.asm`). The assembly program contains the compute logic that runs on the IPU. See the [Assembly Syntax Guide](assembly-syntax.md) for details on writing IPU assembly code with Jinja2 preprocessing.

## Step 2: Define Bazel Build Targets

Create `BUILD.bazel` entries to assemble the program and define your app targets:

```starlark
load("//:asm_rules.bzl", "assemble_asm")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_python_pytest//python_pytest:defs.bzl", "py_pytest_test")

# Assemble the .asm file to .bin
assemble_asm(
    name = "assemble_my_app",
    src = "src/ipu_apps/my_app/my_app.asm",
)

# Collect test data files
filegroup(
    name = "my_app_test_data",
    srcs = glob(["src/ipu_apps/my_app/test_data/**/*.bin"]),
)

# Binary for interactive debugging (optional)
py_binary(
    name = "my_app",
    srcs = ["src/ipu_apps/my_app/__main__.py"],
    main = "src/ipu_apps/my_app/__main__.py",
    data = [
        ":assemble_my_app.bin",
        ":my_app_test_data",
    ],
    env = {
        "MY_APP_INST_BIN": "$(rootpath :assemble_my_app.bin)",
        "MY_APP_DATA_DIR": "src/tools/ipu-apps/src/ipu_apps/my_app/test_data",
    },
    imports = ["src"],
    deps = [":ipu_apps_lib"],
)

# Regression tests
py_pytest_test(
    name = "test_my_app",
    srcs = ["test/test_my_app.py"],
    data = [
        ":assemble_my_app.bin",
        ":my_app_test_data",
    ],
    env = {
        "MY_APP_INST_BIN": "$(rootpath :assemble_my_app.bin)",
        "MY_APP_DATA_DIR": "src/tools/ipu-apps/src/ipu_apps/my_app/test_data",
    },
    imports = ["src"],
    deps = [
        ":ipu_apps_lib",
        requirement("pytest"),
    ],
)
```

Build the assembly:

```bash
bazel build //src/tools/ipu-apps:assemble_my_app
```

## Step 3: Write the Application Class

Create `src/ipu_apps/my_app/__init__.py` and subclass `IpuApp`:

```python
"""My IPU application — description of what it does."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import load_binary_to_xmem, dump_xmem_to_binary
from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState


class MyApp(IpuApp):
    """My application harness.
    
    Args:
        inst_path:    Path to assembled instruction binary.
        inputs_path:  Path to input data binary.
        output_path:  Optional path to write output.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(self.inputs_path)

    def setup(self, state: "IpuState") -> None:
        """Load data into XMEM and configure registers before execution."""
        # Load input data
        load_binary_to_xmem(
            state, self.inputs_path,
            base_addr=0x0000,
            chunk_size=128,
            num_chunks=10,
        )
        
        # Set control registers
        state.regfile.set_cr(0, 0x0000)   # input base address
        state.regfile.set_cr(1, 0x20000)  # weight base address
        state.regfile.set_cr(2, 0x40000)  # output base address

    def teardown(self, state: "IpuState") -> None:
        """Dump results from XMEM after execution."""
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                base_addr=0x40000,
                chunk_size=256,
                num_chunks=10,
            )
```

**Key points:**
- Only implement `setup()` and `teardown()` — the base class handles everything else
- Use `**kwargs` in `__init__` and call `super().__init__(**kwargs)` to auto-store all parameters as attributes
- `setup()` prepares the IPU state before execution (load data, set registers)
- `teardown()` collects results after execution (dump outputs)
- The base class `run()` method orchestrates: create state → load program → setup → execute → teardown

## Step 4: Write the Interactive Runner (Optional)

Create `src/ipu_apps/my_app/__main__.py` for interactive debugging:

```python
"""Debug runner for my_app.

Usage::
    bazel run //src/tools/ipu-apps:my_app -- --input data.bin
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt
from ipu_apps.my_app import MyApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run my_app with debug CLI")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=1_000_000)
    args = parser.parse_args()

    # Bazel sets these via env vars (see BUILD.bazel)
    inst_path = Path(os.environ["MY_APP_INST_BIN"])
    
    app = MyApp(
        inst_path=inst_path,
        inputs_path=args.input,
        output_path=args.output,
    )
    state, cycles = app.run(
        max_cycles=args.max_cycles,
        debug_callback=debug_prompt,  # Interactive debug CLI
    )
    print(f"Finished in {cycles} cycles")


if __name__ == "__main__":
    main()
```

**Key points:**
- Use `debug_prompt` as the callback to enable interactive debugging
- Read `MY_APP_INST_BIN` from environment (set by Bazel in BUILD.bazel)
- Accept command-line arguments for input/output paths
- This is optional — only needed if you want a standalone debug runner

Run interactively:

```bash
bazel run //src/tools/ipu-apps:my_app -- --input my_input.bin --output results.bin
```

## Step 5: Write Regression Tests

Create `test/test_my_app.py` for automated validation:

```python
"""End-to-end regression tests for my_app."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.my_app import MyApp


# Bazel sets these via env vars (see BUILD.bazel)
_INST_BIN = Path(os.environ["MY_APP_INST_BIN"])
_DATA_DIR = Path(os.environ["MY_APP_DATA_DIR"])


def _run_app(tmp_path: Path, test_case: str) -> tuple[bytes, int]:
    """Run the app for a test case, return (output_bytes, cycles)."""
    case_dir = _DATA_DIR / test_case
    if not case_dir.exists():
        pytest.skip(f"Test data not found: {case_dir}")
    
    inputs = case_dir / "inputs.bin"
    if not inputs.exists():
        pytest.skip(f"Missing input file: {inputs}")
    
    output = tmp_path / "output.bin"
    app = MyApp(
        inst_path=_INST_BIN,
        inputs_path=inputs,
        output_path=output,
    )
    _, cycles = app.run(max_cycles=1_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("test_case,golden_name", [
    ("case1", "golden_case1.bin"),
    ("case2", "golden_case2.bin"),
])
def test_my_app(tmp_path: Path, test_case: str, golden_name: str) -> None:
    """Compare app output against golden reference."""
    actual, cycles = _run_app(tmp_path, test_case)
    assert cycles > 0
    
    golden = _DATA_DIR / test_case / golden_name
    if golden.exists():
        assert actual == golden.read_bytes()
```

**Key points:**
- Read `MY_APP_INST_BIN` and `MY_APP_DATA_DIR` from environment variables
- Use `pytest.mark.parametrize` to test multiple cases with one test function
- Compare actual output against golden reference files
- Skip gracefully if test data is missing
- No `debug_callback` — tests run headless and fast

Run tests:

```bash
bazel test //src/tools/ipu-apps:test_my_app
```

## Example Application: Fully Connected Layer

This section walks through a complete real-world implementation: a fully-connected neural network layer that processes multiple samples, each with 128 input neurons and produces 64 output neurons.

### Assembly Program

The IPU assembly implements the core computation: for each input sample, compute the dot product of the input vector with each weight row (after transposition for efficiency):

```asm
set                 lr0 0 ;;
set                 lr1 1280 ;;
set                 lr2 0 ;;

input_loop:
    reset_acc;;

    ldr_cyclic_mult_reg lr0 cr0 lr15;;

    set                 lr4 -128;;
    set                 lr5 -1;;
    set                 lr6 127;;

element_loop:
    ldr_mult_reg        mem_bypass lr4 cr1;
    incr                lr4 128;
    incr                lr5 1;
    mult.ev             mem_bypass lr5 lr15 lr15;
    acc;
    blt                 lr5 lr6 element_loop;;

    str_acc_reg         lr7 cr2;;
    incr                lr7 256;
    incr                lr0 128;;

    break;;

    blt                 lr0 lr1 input_loop;;

end:
    bkpt;;
```

### Python Application Class

The `FullyConnectedApp` class handles data setup and result collection:

```python
"""Fully-connected layer test harness — Python port of fully_connected.c."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import load_binary_to_xmem, dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# Constants
SAMPLES_NUM = 10
INPUT_BASE_ADDR = 0x0000
INPUT_NEURONS = 128
WEIGHTS_BASE_ADDR = 0x20000
OUTPUT_BASE_ADDR = 0x40000
OUTPUT_NEURONS = 64

def parse_dtype(dtype_str: str) -> DType:
    """Parse a dtype string into a DType enum value."""
    dtype_map = {
        "INT8": DType.INT8,
        "FP8_E4M3": DType.FP8_E4M3,
        "FP8_E5M2": DType.FP8_E5M2,
    }
    dt = dtype_map.get(dtype_str)
    if dt is None:
        raise ValueError(
            f"Invalid dtype '{dtype_str}'. Supported: INT8, FP8_E4M3, FP8_E5M2"
        )
    return dt

def _load_and_transpose_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load weights from file and transpose into XMEM.

    Original: (OUTPUT_NEURONS × INPUT_NEURONS).
    Transposed: (INPUT_NEURONS × INPUT_NEURONS), zero-padded.
    """
    raw = Path(weights_path).read_bytes()
    expected = OUTPUT_NEURONS * INPUT_NEURONS
    if len(raw) < expected:
        raise ValueError(
            f"Weights file too small: {len(raw)} bytes, expected {expected}"
        )

    original: list[bytes] = []
    for j in range(OUTPUT_NEURONS):
        row_start = j * INPUT_NEURONS
        original.append(raw[row_start : row_start + INPUT_NEURONS])

    for i in range(INPUT_NEURONS):
        transposed_vector = bytearray(INPUT_NEURONS)
        for j in range(OUTPUT_NEURONS):
            transposed_vector[j] = original[j][i]
        state.xmem.write_address(WEIGHTS_BASE_ADDR + i * INPUT_NEURONS, transposed_vector)


class FullyConnectedApp(IpuApp):
    """Fully-connected layer application harness.

    Args:
        inst_path:    Path to assembled instruction binary.
        inputs_path:  Path to input activations binary.
        weights_path: Path to weights binary.
        output_path:  Optional path to write output.
        dtype:        Data type string or DType.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(self.inputs_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        """Load inputs and weights, set control registers."""
        state.set_cr_dtype(int(self.dtype))
        load_binary_to_xmem(
            state, self.inputs_path, INPUT_BASE_ADDR, INPUT_NEURONS, SAMPLES_NUM
        )
        _load_and_transpose_weights(state, self.weights_path)
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, WEIGHTS_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)

    def teardown(self, state: "IpuState") -> None:
        """Dump output activations from XMEM."""
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_NEURONS * 4, SAMPLES_NUM,
            )
```

### Interactive Debug Runner

The `__main__.py` provides an interactive CLI for development:

```python
"""Debug runner for the fully-connected app.

Usage::

    bazel run //src/tools/ipu-apps:fully_connected -- --dtype INT8
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.fully_connected import FullyConnectedApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fully-connected with debug CLI")
    parser.add_argument("--dtype", default="INT8", choices=["INT8", "FP8_E4M3", "FP8_E5M2"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    # Bazel sets these via env vars (see BUILD.bazel)
    inst_path = Path(os.environ["FC_INST_BIN"])
    data_dir = Path(os.environ["FC_DATA_DIR"])

    app = FullyConnectedApp(
        inst_path=inst_path,
        inputs_path=data_dir / args.dtype.lower() / f"inputs_{args.dtype.lower()}.bin",
        weights_path=data_dir / args.dtype.lower() / f"weights_{args.dtype.lower()}.bin",
        output_path=args.output,
        dtype=args.dtype,
    )
    state, cycles = app.run(max_cycles=args.max_cycles, debug_callback=debug_prompt)
    print(f"Finished in {cycles} cycles")


if __name__ == "__main__":
    main()
```

**Key observations:**
- The `setup()` method transcodes weights (transpose for cache efficiency) and loads both inputs and weights into XMEM
- Control registers point the assembly code to the base addresses: `CR0=inputs`, `CR1=weights_transposed`, `CR2=outputs`
- The dtype is configurable (INT8, FP8_E4M3, FP8_E5M2) and passed to the IPU state
- The assembly program is written to operate on transposed weights for better performance
- All paths are resolved by Bazel and passed via environment variables

See [src/tools/ipu-apps/src/ipu_apps/fully_connected](https://github.com/rechefe/ipu-emulator/tree/master/src/tools/ipu-apps/src/ipu_apps/fully_connected) for the complete working code including tests and Bazel build targets.

## Summary: The Two Usage Patterns

### Pattern 1: Interactive Debugging
- **Purpose**: Develop and debug your application
- **File**: `__main__.py` with `debug_prompt` callback
- **Target**: `py_binary` in BUILD.bazel
- **Run**: `bazel run //...:my_app -- --input data.bin`
- **Features**: Breakpoints, register inspection, step-through execution

### Pattern 2: Regression Tests
- **Purpose**: Validate correctness against known-good outputs
- **File**: `test/test_*.py` with pytest assertions
- **Target**: `py_pytest_test` in BUILD.bazel
- **Run**: `bazel test //...:test_my_app`
- **Features**: Fast, headless, CI-friendly, parametrized test cases

Both patterns share the same `MyApp` class — you just write `setup()` and `teardown()` once, then use it in both contexts.

## Key Concepts

- **Memory Layout**: Define base addresses for inputs, weights, and outputs in external memory (XMEM)
- **Register Setup**: Initialize LR/CR registers before execution in `setup()`
- **Auto-attribute Storage**: Pass all parameters to `IpuApp.__init__(**kwargs)` — they're automatically stored as `self.param_name`
- **Path Handling**: Use Bazel `env` with `$(rootpath ...)` to set environment variables with resolved paths
- **Emulator Run**: The emulator executes instructions until the program counter exceeds instruction memory
- **Bazel Integration**: The `assemble_asm` rule compiles `.asm` files to `.bin` executables

See the [Assembly Syntax Guide](assembly-syntax.md) for more details on writing IPU programs and the complete fully_connected example at `src/tools/ipu-apps/src/ipu_apps/fully_connected/` for a real-world implementation.
