# Building IPU Applications

This guide shows how to build a complete IPU application using the fully connected neural network layer as an example. The complete code is in [src/tools/ipu-apps/src/ipu_apps/fully_connected](https://github.com/rechefe/ipu-emulator/tree/master/src/tools/ipu-apps/src/ipu_apps/fully_connected).

## Application Structure

Each IPU application is a subpackage under `ipu_apps/` containing:

1. **Assembly code** (`.asm`) — IPU program with compute operations (see [Assembly Syntax Guide](assembly-syntax.md))
2. **Python app class** (`__init__.py`) — Subclass of `IpuApp` that implements `setup()` and `teardown()`
3. **Reusable cases** (`cases.py`) — Input preparation and output checks without pytest
4. **Test data** — Input/output binary files for validation
5. **Regression tests** (`test.py`) — Tests importing the reusable cases

Everything lives together in one directory.

## Configure the IPU before execution

Application setup is responsible for loading data and selecting the IPU
configuration that the assembly program reads. Keep this in the Python
`setup()` hook so assembly remains focused on compute instructions. See
[IPU Configuration](ipu-configuration.md) for the full register layout.

```python
from ipu_emu.ipu_config import LR_CR_SCALAR_VALUE_MASK
from ipu_emu.ipu_math import DType

def setup(self, state: IpuState) -> None:
    # dtype is emulator-only state, not a CR register.
    state.dtype = DType.INT8

    # CR15 dstructure: configure whichever CR register your AGG.*/ACTIVATE/AAQ
    # instructions name via their mandatory cr_idx operand.
    state.set_cr_dstructure(valid_elements=128, partition=0)

    # CR0 and CR1 are read-only constants (0 and 1). Use CR2-CR14 for app data.
    state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
    state.regfile.set_cr(3, 128)  # stride
    state.regfile.set_cr(13, WEIGHTS_BASE_ADDR)

    # LR/CR values are 32-bit scalars; mask wrapped constants explicitly.
    state.regfile.set_cr(9, (-128) & LR_CR_SCALAR_VALUE_MASK)
```

In assembly, every `AGG.*`/`ACTIVATE`/`AAQ` instruction must name its
dstructure CR register explicitly via a mandatory `cr_idx` operand — there is
no implicit default:

```asm
AGG.SUM LR0, CR15;;
ACTIVATE relu, CR15;;
AGG.SUM LR0, CR3;;
ACTIVATE relu, CR3;;
```

For aggregation (`AGG.SUM`, `AGG.MAX`, etc.) the element count is controlled by the required `cr_idx` operand — it reads `valid_elements` from the named CR register. `CR15` remains a valid choice (it is the dstructure register's conventional home), but it must be written out like any other register.

## Wide-vector debug mode (optional)

The emulator can run multiply/accumulate paths with **128×32-bit elements** (FP32 or INT32) instead of 8-bit vectors, for debugging without quantization on that path. XMEM addresses stay the same; load sizes and alignment rules change. See **[Wide-vector debug mode](wide-vector-debug-mode.md)** for how to construct `IpuState`, prepare 512-byte loads, and use `AAQ` / **`STR_POST_AAQ_REG`** / `STR_ACC_REG` in that mode.

## Activations, `ACTIVATE`, and virtual α (Python emulator) {#activations-emulator}

The [AaQ and Store stage spec](specs/stage-aaq-str.md) describes how **real hardware** wires activation: a function id (for example from `act_cr_idx` and a `CR` read) and **α-like parameters** that are **not** VLIW immediates—they come from implementation-defined configuration (constants, fuses, side-band registers, etc.).

The **Python emulator** in this repository adds a convenience AAQ-slot instruction **`ACTIVATE`** so programs can apply the same nine activation shapes to elements read from **`R_ACC`**, writing results into **`POST_AAQ_REG`** (without modifying **`R_ACC`**), without modeling the full `act_cr_idx` path:

```asm
ACTIVATE relu, CR15;;
```

- **Syntax:** `ACTIVATE activation_fn, cr_idx`, where *activation_fn* is a **keyword** (`identity`, `relu`, `relu6`, `sigmoid`, `tanh`, `gelu`, `softplus`, `elu`, `exp2`). The active element count comes from `cr_idx`'s `valid_elements`, the same dstructure field used by the `AGG.*` instructions; `cr_idx` is mandatory (any `CR0`–`CR15`, no implicit default).
- **Single source of truth:** keyword order and the pure-Python math live in `src/tools/ipu-common/src/ipu_common/activations.py` (`ACTIVATION_FN_NAMES`, `apply_activation`).

### `R_ACC`, `POST_AAQ_REG`, and `STR_POST_AAQ_REG` (staging vs export)

- **Accumulation** stays in **`R_ACC`** (512 bytes = 128×32-bit elements). The ACC-slot **`AGG.SUM`** / **`AGG.SUM.FIRST`** / **`AGG.MAX`** / **`AGG.MAX.FIRST`** reduce **`R_ACC`** in place, writing a single `R_ACC` slot selected by an LR register; hardware uses the `act_cr_idx` path described in the AAQ spec for activation selection.
- **`ACTIVATE`** (emulator) reads **`R_ACC`** and writes element-wise **32→32** activated elements into **`POST_AAQ_REG`**; **`R_ACC`** is left unchanged.
- **`POST_AAQ_REG`** is **temporarily a 512-byte** wide staging register (same element layout as **`R_ACC`**) until end-to-end quantization and export are finalized.
- **`AAQ`** (INT8 mode) quantizes the **wide elements currently in `POST_AAQ_REG`**, writing **128 bytes** of clamped INT8 into the **leading** bytes of **`POST_AAQ_REG`** and clearing the remainder for now. Typical flow: **`ACTIVATE`** (wide elements) then **`AAQ`** (quantize in place into the same register’s byte prefix).
- **`STR_POST_AAQ_REG`** stores **`POST_AAQ_REG`** — **512 bytes** — to XMEM (whatever wide or quantized layout that buffer holds at issue time).

### Virtual α in the emulator (elu)

The stock ISA does not expose α. The emulator reads α from each **`IpuState`** (field `elu_alpha`). If you do not set it, it is initialized from the default float in `ipu_common/activations.py` (`DEFAULT_ELU_ALPHA`).

Configure α the same way you think about dtype on state — **not via CR**:

```python
from ipu_emu.ipu_state import IpuState
from ipu_emu.emulator import run_test

state = IpuState(elu_alpha=1.0)
# or after construction:
state.set_activation_alphas(elu_alpha=1.0)

# High-level harness (optional kwargs mirror IpuState):
state, cycles = run_test(
    inst_path="prog.bin",
    setup=my_setup,
    elu_alpha=1.0,
)
```

Subclassing **`IpuApp`**, you can pass α through **`run(...)`** or store them on the app from **`__init__`** (same names as `run_test`); explicit **`run()`** arguments override stored attributes:

```python
app = MyApp(inst_path="prog.bin", elu_alpha=1.0)
state, cycles = app.run()  # uses 1.0
state, cycles = app.run(elu_alpha=0.5)  # uses 0.5 for this run
```

Use a **fresh `IpuState`** (or a new `run_test` call with different α kwargs) when you need different α values in the same process.

## Step 1: Write the Assembly Program

Create your IPU assembly program (e.g., `fully_connected.asm`). The assembly program contains the compute logic that runs on the IPU. See the [Assembly Syntax Guide](assembly-syntax.md) for details on writing IPU assembly code with Jinja2 preprocessing.

## Step 2: Define Bazel Build Targets

Add one declaration to `src/tools/ipu-apps/BUILD.bazel`:

```starlark
load("//:ipu_app.bzl", "ipu_app")

ipu_app(
    name = "my_app",
    kernel_package = "src/ipu_apps/my_app",
    deps = [":ipu_apps_lib"],
    test_deps = [requirement("pytest")],
    data = glob(["src/ipu_apps/my_app/test_data/**/*.bin"]),
)
```

The macro supplies the shared runner and a `test_my_app` target. It packages
`my_app.asm`; if `SPEC.asm` uses a different filename, pass the same relative
path as the macro's `asm` argument. Cases assemble the source at runtime, so no
instruction-path or fixture-directory environment variables are needed.

```bash
bazel run //src/tools/ipu-apps:my_app -- --list-cases
bazel test //src/tools/ipu-apps:test_my_app
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

## Step 4: Declare the Spec and Reusable Cases

Declare a `KernelSpec` named `SPEC` beside your harness. It defines the
supported parameters, constructor arguments, assembly path, and execution
configuration. Follow [Adding applications](adding-applications.md) for the
spec and constructor-guard contract.

Create `src/ipu_apps/my_app/cases.py` for input preparation and output checks:

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
`my_app/app.py` therefore uses `my_app/cases.py`. Set `SPEC.package` explicitly
when resources live in a different package.

## Step 5: Write Regression Tests

Create `src/ipu_apps/my_app/test.py`, importing the runtime cases:

```python
import pytest
from ipu_apps.my_app.cases import CASES
from ipu_apps.kernel_registry.cases import run_case


@pytest.mark.parametrize("name", CASES)
def test_my_app(name):
    state, cycles = run_case("my_app", CASES[name])
    assert state.is_halted and cycles > 0
```

`run_case` assembles, prepares, executes, and validates the case, then cleans
its temporary workspace. Tests may pass `inst_path` from a module-scoped
fixture to reuse assembly. Use `workspace` when a test needs to inspect files.

```bash
bazel test //src/tools/ipu-apps:test_my_app
```

The pyproject config discovers both `test/test_*.py` and adjacent `src/**/test.py`
suites. Bazel remains the supported build and test workflow.

## Inspecting Run Statistics

Every run populates `state.stats` (a `RunStats` instance) with counters useful for spotting bottlenecks. They are updated automatically by `run_until_complete` / `run_with_debug` — no opt-in flag is required.

Available fields:

- `total_cycles` — VLIW cycles executed
- `mult_active_cycles` — cycles whose MULT slot was not `MULT_NOP`
- `acc_active_cycles` — cycles whose ACC slot was not `ACC_NOP`
- `xmem_reads` — count of `LDR_MULT_REG`, `LDR_CYCLIC_MULT_REG`, `LDR_MULT_MASK_REG`
- `xmem_writes` — count of `STR_ACC_REG`, `STR_POST_AAQ_REG`
- `xmem_accesses` — sum of reads + writes
- `mult_utilization`, `acc_utilization` — active-cycle fraction (0.0–1.0)

`state.stats.format_summary()` returns a ready-to-print multi-line block:

```text
=== Run summary ===
Total cycles:           12847
Mult active:            10421  (81.1%)
Acc  active:             9876  (76.9%)
XMEM reads:              3200
XMEM writes:              128
XMEM accesses:           3328
```

Typical usage from an interactive runner or a test:

```python
state, cycles = app.run(max_cycles=args.max_cycles)
print(state.stats.format_summary())

# Or pick individual fields for assertions / metrics:
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
    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr4 cr9 ;;
    SET                 lr5 cr10 ;;
    SET                 lr6 cr11 ;;
    SET                 lr15 cr12 ;;

    LDR_CYCLIC_MULT_REG lr4 cr1 lr15;
    ADD                 lr4 lr4 cr3;
    ADD                 lr5 lr5 cr4;
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.FIRST;;
    BNE                 lr5 lr6 element_loop;;
    B                   after_element_loop;;

element_loop:
    LDR_CYCLIC_MULT_REG lr4 cr1 lr15;
    ADD                 lr4 lr4 cr3;
    ADD                 lr5 lr5 cr4;
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC;;
    BNE                 lr5 lr6 element_loop;;

after_element_loop:
    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 cr5;
    ADD                 lr0 lr0 cr3;;

    BREAK;;

    BLT                 lr0 lr1 input_loop;;

end:
    BKPT;;
```

### Python Application Class

The `FullyConnectedApp` class handles data setup and result collection:

```python
"""Fully-connected layer test harness — Python port of fully_connected.c."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_config import LR_CR_SCALAR_VALUE_MASK
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
        "FP8_E4M3": DType.E4,
        "FP8_E5M2": DType.E5,
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
        state.dtype = self.dtype
        state.set_cr_dstructure(valid_elements=INPUT_NEURONS, partition=0)
        load_binary_to_xmem(
            state, self.inputs_path, INPUT_BASE_ADDR, INPUT_NEURONS, SAMPLES_NUM
        )
        _load_and_transpose_weights(state, self.weights_path)
        # CR0 is permanently 0; CR1 is permanently 1.
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, 128)
        state.regfile.set_cr(4, 1)
        state.regfile.set_cr(5, 256)
        # Values for ``SET lr* cr*`` in the assembly listing above
        state.regfile.set_cr(6, 0)
        state.regfile.set_cr(7, 1280)
        state.regfile.set_cr(8, 0)
        state.regfile.set_cr(9, (-128) & LR_CR_SCALAR_VALUE_MASK)
        state.regfile.set_cr(10, (-1) & LR_CR_SCALAR_VALUE_MASK)
        state.regfile.set_cr(11, 127)
        state.regfile.set_cr(12, 0)

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
bazel test //src/tools/ipu-apps:test_fully_connected
```

Fixture paths are resolved relative to the case module. `--output` exports
completed output before validation, allowing inspection of a failing result;
a failed check still exits with a nonzero status and a diagnostic. Programs
that exceed their cycle limit are rejected before export.

The shared CLI runs without an interactive debugger. To debug a harness
through the Python API, pass `debug_callback=debug_prompt` to `app.run()`;
see [Debugging](debugging.md) for callback usage.

## Running and Testing Applications

Both `bazel run` and `bazel test` use the same harness and reusable cases.
The `ipu_app` macro supplies the CLI, and the adjacent `test.py` adds pytest
coverage. Kernel packages do not need a custom `__main__.py`.

## Key Concepts

- **Memory Layout**: Define base addresses for inputs, weights, and outputs in external memory (XMEM)
- **Register Setup**: Initialize LR/CR registers before execution in `setup()`
- **Auto-attribute Storage**: Pass all parameters to `IpuApp.__init__(**kwargs)` — they're automatically stored as `self.param_name`
- **Path Handling**: Resolve fixtures relative to `cases.py`; declare assembly and fixture files as Bazel data
- **Emulator Run**: The emulator executes instructions until the program counter exceeds instruction memory
- **Bazel Integration**: `ipu_app` supplies run and test targets; `assemble_asm` remains available for standalone binary artifacts

See the [Assembly Syntax Guide](assembly-syntax.md) for more details on writing IPU programs and the complete fully_connected example at `src/tools/ipu-apps/src/ipu_apps/fully_connected/` for a real-world implementation.
