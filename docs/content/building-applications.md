# Building IPU Applications

!!! note "Legacy: the `ipu_apps` kernel framework"
    This page describes kernels under `src/tools/ipu-apps`, written as an
    `IpuApp` subclass that programs the CRs by hand. New kernels go under
    `kernels/` with a `kernel.toml` and no harness: see
    [Write a kernel](how-to/write-a-kernel.md) and the
    [`kernel.toml` reference](reference/kernel-toml.md). E07 migrates the
    kernels below and then removes this page.

This guide shows how to build a complete IPU application using the fully connected neural network layer as an example. The complete code is in [src/tools/ipu-apps/src/ipu_apps/kernels/linear/fully_connected](https://github.com/rechefe/ipu-emulator/tree/master/src/tools/ipu-apps/src/ipu_apps/kernels/linear/fully_connected).

## Application Structure

Each IPU application is its own folder, `ipu_apps/kernels/<family>/<app>/`, containing:

1. **Assembly code** (`.asm`) — IPU program with compute operations (see [Assembly Syntax Guide](reference/isa/assembly-syntax.md))
2. **Python app class** (`app.py`) — Subclass of `IpuApp` that implements `setup()` and `teardown()`, with `SPEC` at the bottom
3. **Reusable cases** (`cases.py`) — Input preparation and output checks without pytest
4. **Test data** — Input/output binary files for validation
5. **Regression tests** (`test.py`) — runs every case in `cases.py` via one shared line, plus any kernel-specific tests

Everything lives together in one directory. `__init__.py` stays empty.

## Configure the IPU before execution

Application setup is responsible for loading data and selecting the IPU
configuration that the assembly program reads. Keep this in the Python
`setup()` hook so assembly remains focused on compute instructions. See
[Configuration registers](reference/isa/configuration-registers.md) for the full register layout.

```python
from ipu_isa.constants import LR_CR_SCALAR_VALUE_MASK

def setup(self, state: IpuState) -> None:
    # The element format (state.fmt) is fixed when the state is built
    # (SPEC.execution); it is emulator configuration, not a CR register.

    # CR15 dstructure: configure whichever CR register your AGG.*/ACTIVATE.QUANTIZE
    # instructions name via their mandatory cr_idx operand.
    state.set_cr_dstructure(valid_elements=128, partition=0)

    # CR0 and CR1 are read-only constants (0 and 1). Use CR2-CR14 for app data.
    # XMEM operands in .asm are row numbers (128-byte rows here), not byte addresses.
    state.regfile.set_cr(2, OUTPUT_BASE_ADDR // 128)
    state.regfile.set_cr(3, 1)  # stride: one row
    state.regfile.set_cr(13, WEIGHTS_BASE_ADDR // 128)

    # LR/CR values are 32-bit scalars; mask wrapped constants explicitly.
    state.regfile.set_cr(9, (-1) & LR_CR_SCALAR_VALUE_MASK)
```

Every `AGG.*` and `ACTIVATE.QUANTIZE` instruction names the CR that holds its
dstructure word; [Configuration registers](reference/isa/configuration-registers.md)
describes the word, its fields and `set_cr_dstructure`.

## Element formats

A kernel runs in one element format. [Element formats](explanation/element-formats.md) lists them, what each one changes, and how a kernel selects one.

## Activations and `ACTIVATE.QUANTIZE`

[`ACTIVATE.QUANTIZE` in the emulator](reference/isa/activate-quantize.md) describes the
instruction, `R_ACC` and `POST_AAQ_REG`, and how to set `elu`'s α and the `window`
bounds on the state. An `IpuApp` also takes them as keywords of `__init__` or
`run()` (`elu_alpha`, `window_a`, `window_b`); a `run()` keyword overrides the
stored value for that run.

## Step 1: Write the Assembly Program

Create your IPU assembly program (e.g., `fully_connected.asm`). The assembly program contains the compute logic that runs on the IPU. See the [Assembly Syntax Guide](reference/isa/assembly-syntax.md) for details on writing IPU assembly code with Jinja2 preprocessing. To assemble a template from the command line, give each parameter its `ipu:params` header declares with `--define`:

```bash
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --input "$PWD/kernel.asm" --output "$PWD/kernel.bin" --define rows=8
```

The `format` subcommand prints a source in the repository's layout; `--check` exits 1 when the source would change, for CI and pre-commit hooks. [Command-line tools](reference/cli.md#assembler-ipu-as) lists every `ipu-as` subcommand and option, including `check`, `disassemble`, `--target hardware` and the `format` exit codes:

```bash
bazel run //src/tools/ipu-as-py:ipu-as -- format --check --input "$PWD/kernel.asm"
```

## Step 2: Define Bazel Build Targets

Nothing to declare. `src/tools/ipu-apps/BUILD.bazel` globs every `.asm` under
`src/ipu_apps/kernels/`, so `kernels/<family>/my_app/my_app.asm` alone produces:

- `:my_app` — `bazel run` runs the app; `bazel test` runs every case in
  `cases.py` through the required `test.py` (`:test_my_app` remains an alias);
- `:assemble_my_app` — the standalone assembled binary;
- `:benchmark_my_app` — only if a `benchmark.py` sits beside the `.asm`. It
  declares `CONFIGS`, a list of overrides of the default case's options
  (optionally `PER = "rows"` for a cycles-per-row column, and `MAX_CYCLES`);
  the shared runner in `kernel_registry/benchmarking.py` runs each through the
  case and writes `results.md` beside it.

`.bin` fixtures in the package are included as data.
Cases assemble the source at runtime, so no instruction-path or
fixture-directory environment variables are needed.
Each kernel has its own directory with one `.asm`, one `SPEC` and one `CASES`;
kernels that share code keep it in their family's `app.py`/`cases.py`.

```bash
bazel run //src/tools/ipu-apps:my_app -- --list-cases
bazel test //src/tools/ipu-apps:my_app
```

## Step 3: Write the Application Class

Create `src/ipu_apps/kernels/<family>/my_app/app.py` and subclass `IpuApp`:

```python
"""My IPU application — description of what it does."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import load_binary_to_xmem, dump_xmem_to_binary
from ipu_apps.kernel_registry.base import IpuApp

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
            max_chunks=10,
        )

        # Set control registers. CR0/CR1 are read-only constants (0 and 1), and
        # .asm XMEM operands are row numbers (128-byte rows in narrow mode),
        # not byte addresses.
        state.regfile.set_cr(2, 0x0000 // 128)   # input base row
        state.regfile.set_cr(3, 0x20000 // 128)  # weight base row
        state.regfile.set_cr(4, 0x40000 // 128)  # output base row

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

## Step 4: Declare the Spec and Reusable Cases

Declare a `KernelSpec` named `SPEC` beside your harness. It defines the
supported parameters, constructor arguments, assembly path, and execution
configuration. Follow [Adding applications](adding-applications.md) for the
spec and constructor-guard contract.

Create `src/ipu_apps/kernels/<family>/my_app/cases.py` for input preparation and output checks:

```python
from pathlib import Path
from ipu_apps.kernel_registry.cases import (
    KernelCase, PreparedCase, check_output_bytes,
)

DATA = Path(__file__).with_name("test_data")


def prepare(workspace):
    output = workspace / "output.bin"

    def check():
        check_output_bytes(output, DATA / "expected.bin")

    return PreparedCase(
        params={},  # Parameters accepted by your SPEC.
        bindings={"inputs_path": DATA / "inputs.bin", "output_path": output},
        check=check,
    )


CASES = {"default": KernelCase(prepare)}
```

Cases must not import pytest. A checker raises a descriptive exception on a
mismatch; use explicit comparisons rather than `assert`, which disappears
under Python optimization. Numeric checks should report the error and tolerance.

Case defaults define CLI options and must be strings, integers, floats, or
booleans. Option names use lower-case letters, digits, and underscores and may
not conflict with the shared runner's options. For richer inputs, accept a
string and parse it in preparation. A `default` case is required.

Cases and assembly live in the harness's containing package. A class in
`kernels/<family>/my_app/app.py` therefore uses `my_app/cases.py`. Set `SPEC.package` explicitly
when resources live in a different package.

## Step 5: Write Regression Tests

Create `kernels/<family>/my_app/test.py`. Its first line of code runs every
case in `cases.py` through `run_case`, which assembles, prepares, executes and
validates it, then cleans its temporary workspace. Below it, add checks beyond
the cases — edge-case geometry, refusal messages, extra numerical properties:

```python
from ipu_apps.kernel_registry import resolve
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__)


def test_refuses_odd_width():
    assert not resolve("my_op", shape=(1, 3)).supported
```

To run the default case at more sizes, pass option overrides instead of writing
a parametrized test: `case_tests(__package__, sweep=[dict(rows=8), dict(rows=300)])`.

Tests may pass `inst_path` from a module-scoped fixture to reuse assembly. Use
`workspace` when a test needs to inspect files.

`ST.ACC` is emulator-only: every execution emits
`ipu_emu.errors.DebugOnlyInstructionWarning`. `case_tests` reads the assembled
program and, when it contains the instruction, requires the warning in every
case; for any other kernel the warning is an error. A test that calls
`run_case` or `app.run` itself wraps the run in
`pytest.warns(DebugOnlyInstructionWarning)`, or in
`debug_only_warning(kernel_uses_debug_only_instructions(kernel))` from
`ipu_apps.kernel_registry.testing`.

```bash
bazel test //src/tools/ipu-apps:my_app
```

The pyproject config discovers both `test/test_*.py` and adjacent `src/**/test.py`
suites. Bazel remains the supported build and test workflow.

## Inspecting Run Statistics

Every run fills `state.stats`, and `state.stats.format_summary()` prints them.
[Run statistics](reference/cli.md#run-statistics) explains each counter, when
the counters reset and the cycle limit.

```python
state, cycles = app.run(max_cycles=args.max_cycles)
print(state.stats.format_summary())
assert state.stats.mult_utilization > 0.5
```

## Example Application: Fully Connected Layer

This section walks through a complete real-world implementation: a fully-connected neural network layer that processes multiple samples, each with 128 input neurons and produces 64 output neurons.

### Assembly Program

The IPU assembly implements the core computation: activations for the current sample live in **`r0`** (loaded once per sample). Each inner-loop iteration loads a 128-byte **weight row** into the cyclic register (**`r_cyclic`**) and issues **`MULT.RC.VE`**, which multiplies that row by the scalar **`r0[lr5]`** (loop counter advanced via **`ADD`**), then accumulates. The trailing **`cr15`** operand on **`MULT.RC.VE`** names the dstructure register supplying `partition` for element masking — every masking multiply instruction must name a CR register explicitly, with no implicit default. The harness initializes **`cr3`**, **`cr4`**, and **`cr5`** with stride constants **128**, **1**, and **256** so the program can add large steps without the removed **`incr`** mnemonic.

```asm
    SET                 lr0 cr6 ;;
    SET                 lr1 cr7 ;;
    SET                 lr2 cr8 ;;

input_loop:
    LD.MULT        r0 lr0 cr0;;

    SET                 lr4 cr9 ;;
    SET                 lr5 cr10 ;;
    SET                 lr6 cr11 ;;
    SET                 lr15 cr12 ;;

    LD.MULT.CYCLIC lr4 cr1 lr15;
    ADD                 lr4 lr4 cr3;
    ADD                 lr5 lr5 cr4;
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD.FIRST;;
    BNE                 lr5 lr6 element_loop;;
    B                   after_element_loop;;

element_loop:
    LD.MULT.CYCLIC lr4 cr1 lr15;
    ADD                 lr4 lr4 cr3;
    ADD                 lr5 lr5 cr4;
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD;;
    BNE                 lr5 lr6 element_loop;;

after_element_loop:
    ST.ACC              lr7 cr2;;
    ADD                 lr7 lr7 cr5;
    ADD                 lr0 lr0 cr3;;

    BREAK;;

    BLT.S               lr0 lr1 input_loop;;

end:
    HALT;;
```

### Python Application Class

The `FullyConnectedApp` class handles data setup and result collection:

```python
"""Fully-connected layer test harness — Python port of fully_connected.c."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_isa.constants import LR_CR_SCALAR_VALUE_MASK
from ipu_isa.formats import ElementFormat
from ipu_emu.emulator import load_binary_to_xmem, dump_xmem_to_binary

from ipu_apps.kernel_registry.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# Constants
SAMPLES_NUM = 10
INPUT_BASE_ADDR = 0x0000
INPUT_NEURONS = 128
WEIGHTS_BASE_ADDR = 0x20000
OUTPUT_BASE_ADDR = 0x40000
OUTPUT_NEURONS = 64

def parse_dtype(dtype_str: str) -> ElementFormat:
    """Parse a dtype string into the element format it names."""
    dtype_map = {
        "INT8": ElementFormat.INT8,
        "FP8_E4M3": ElementFormat.E4,
        "FP8_E5M2": ElementFormat.E5,
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
        dtype:        Data type string or ElementFormat.
    """

    def __init__(self, *, dtype: str | ElementFormat = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs_path = Path(self.inputs_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        """Load inputs and weights, set control registers."""
        # The state is built in self.dtype (SPEC.execution); its format is read-only.
        state.set_cr_dstructure(valid_elements=INPUT_NEURONS, partition=0)
        load_binary_to_xmem(
            state, self.inputs_path, INPUT_BASE_ADDR, INPUT_NEURONS, SAMPLES_NUM
        )
        _load_and_transpose_weights(state, self.weights_path)
        # CR0 is permanently 0; CR1 is permanently 1. XMEM operands are row
        # numbers, so base addresses are divided by the row size.
        row_size = state.format_info.row_bytes
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR // row_size)
        state.regfile.set_cr(3, 1)          # one row per input vector
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, 512 // row_size)
        # Values for ``SET lr* cr*`` in the assembly listing above
        state.regfile.set_cr(6, 0)
        state.regfile.set_cr(7, SAMPLES_NUM)  # loop bound, in rows
        state.regfile.set_cr(8, 0)
        state.regfile.set_cr(9, (-1) & LR_CR_SCALAR_VALUE_MASK)
        state.regfile.set_cr(10, (-1) & LR_CR_SCALAR_VALUE_MASK)
        state.regfile.set_cr(11, 127)
        state.regfile.set_cr(12, 0)
        state.regfile.set_cr(13, WEIGHTS_BASE_ADDR // row_size)

    def teardown(self, state: "IpuState") -> None:
        """Dump output activations from XMEM."""
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_NEURONS * 4, SAMPLES_NUM,
            )
```

### Running the Fully Connected Cases

The shared registry runner loads `fully_connected/cases.py`:

```bash
bazel run //src/tools/ipu-apps:fully_connected -- --list-cases
bazel run //src/tools/ipu-apps:fully_connected -- --dtype INT8 --output /tmp/fc-output.bin
bazel test //src/tools/ipu-apps:fully_connected
```

Fixture paths are resolved relative to the case module. `--output` exports
completed output before validation, allowing inspection of a failing result;
a failed check still exits with a nonzero status and a diagnostic. Programs
that exceed their cycle limit are rejected before export.

To stop a kernel in the debugger, run its target with `--config=debug`; see
[Debug a kernel](how-to/debug-a-kernel.md).

## Running and Testing Applications

Both `bazel run` and `bazel test` use the same harness and reusable cases.
The `ipu_app` macro supplies the CLI, and the adjacent `test.py` runs every
case plus any kernel-specific pytest coverage. Kernel packages do not need a custom `__main__.py`.

## Key Concepts

- **Memory Layout**: Define base addresses for inputs, weights, and outputs in external memory (XMEM)
- **Register Setup**: Initialize LR/CR registers before execution in `setup()`
- **Auto-attribute Storage**: Pass all parameters to `IpuApp.__init__(**kwargs)` — they're automatically stored as `self.param_name`
- **Path Handling**: Resolve fixtures relative to `cases.py`; declare assembly and fixture files as Bazel data
- **Emulator Run**: The emulator executes instructions until the program counter exceeds instruction memory
- **Bazel Integration**: `ipu_app` supplies a single label for run and test, plus `assemble_<name>` for the standalone assembled binary

See the [Assembly Syntax Guide](reference/isa/assembly-syntax.md) for more details on writing IPU programs and the complete fully_connected example at `src/tools/ipu-apps/src/ipu_apps/kernels/linear/fully_connected/` for a real-world implementation.
