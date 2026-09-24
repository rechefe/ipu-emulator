# ipu-apps

IPU application test harnesses — Python ports of the C test harnesses.

## Framework

Subclass `IpuApp`, write `setup` and `teardown`, call `run`:

```python
from ipu_apps.kernel_registry.base import IpuApp

class MyApp(IpuApp):
    def setup(self, state):
        load_binary_to_xmem(state, self.data_path, 0x0000, 128)
        state.regfile.set_cr(2, 0x0000)  # CR0 and CR1 are read-only

    def teardown(self, state):
        if self.output_path:
            dump_xmem_to_binary(state, self.output_path, 0x1000, 128, 1)
```

Extra `__init__` kwargs are stored as attributes automatically:

```python
app = MyApp(inst_path="program.bin", data_path="data.bin", output_path="out.bin")
state, cycles = app.run()
```

## Existing apps

### Fully Connected

Port of `fully_connected.c` — loads inputs/weights, transposes weights,
runs the FC assembly, dumps output activations.

```python
from ipu_apps.kernels.linear.fully_connected.app import FullyConnectedApp

app = FullyConnectedApp(
    inst_path="fc.bin",
    inputs_path="inputs.bin",
    weights_path="weights.bin",
    output_path="output.bin",
    dtype="INT8",
)
state, cycles = app.run()
```

```bash
bazel test //src/tools/ipu-apps:fully_connected
```


## Registered harnesses and reusable cases

Each kernel has an `app.py` (harness hooks and `SPEC`), an empty `__init__.py`,
its assembly, a `cases.py` (runtime inputs and output checks), and a `test.py`
(runs every case via `test_case = case_tests(__package__)`, plus kernel-specific
tests). A family's shared code lives in the family directory under the same
names (`<family>/app.py`, `<family>/cases.py`, `<family>/test.py`), and
`bazel test :<family>` runs the whole family.
Kernel-specific layout and register setup stay in the harness; assembly, construction, execution, and
case checking use the shared registry utilities.

```bash
bazel run //src/tools/ipu-apps:identity
bazel run //src/tools/ipu-apps:softmax_rows_partial -- --rows 8 --n 32
bazel run //src/tools/ipu-apps:identity -- --list-cases
bazel run //src/tools/ipu-apps:identity -- --case single_row
bazel test //src/tools/ipu-apps:identity
```

All 21 assembly kernels have one label supporting both commands:

```bash
bazel run //src/tools/ipu-apps:maxpool2d_stride2
bazel test //src/tools/ipu-apps:maxpool2d_stride2
bazel run --config=debug //src/tools/ipu-apps:maxpool2d_stride2
bazel test //src/tools/ipu-apps:all
```

`bazel test` runs the adjacent `test.py`; `bazel run` runs the selected case.
The old `test_<name>` labels remain compatibility aliases. Tests retain Bazel
XML reports, `--test_filter`, and pytest options supplied with `--test_arg`.
The shared launcher uses Bazel's documented `BUILD_WORKING_DIRECTORY` run
marker to distinguish the commands, and imports pytest only in test mode.
See the [kernel target list](../../../docs/content/debugging.md#quick-start).
Existing softmax
shape, seed, scale, and cycle-limit defaults are preserved. Fully connected's
default INT8 case retains wide arithmetic; its named `int8` and FP8 cases
exercise native arithmetic. `--output PATH` exports completed output before
validation, including failed results for inspection. Validation failures still
exit nonzero.

The registry is the harness factory:

```python
from ipu_apps.kernel_registry import create_harness

app = create_harness(
    "identity",
    params={"shape": (3, 128)},
    bindings={"inst_path": "identity.bin", "input_path": "input.bin",
              "output_path": "output.bin"},
)
state, cycles = app.run()
```

An exact kernel name is validated using its `SPEC`; it never routes to another
implementation. For automatic selection, call `resolve(op, **params)` and pass
the selected kernel name and the same parameters to `create_harness`. The same
query works from the command line for any operation
(`bazel run //src/tools/ipu-apps:query -- softmax shape=32,300 dim=1`; no
arguments prints the coverage report).
Bindings contain file paths and cannot override validated configuration.

`cases.py` declares `CASES`, mapping names to `KernelCase` objects, including
`default`. A case preparation function receives a workspace and its declared
options (str, int, float, or bool defaults) and returns
`PreparedCase(params, bindings, check)`. The checker reads completed output and
raises on a mismatch. `run_case(name, case)` assembles,
constructs through the registry, runs, checks, and cleans temporary files.
Pytest functions use that same path. Importing a case must not execute it.
Cases must not import pytest. Keep pytest imports in `test.py` and use Bazel
for dependency management and testing. Runtime checks must raise descriptive
exceptions rather than rely on assertions.

No BUILD edit is needed: every `src/ipu_apps/kernels/**/<name>.asm` gets
`:<name>` (run + test), `:test_<name>`, and `:assemble_<name>` automatically,
plus `:benchmark_<name>` when a `benchmark.py` sits beside it (`benchmark.py`
declares only `CONFIGS`, overrides of the default case's options, and the
shared runner writes `results.md`). `SPEC.name` and `SPEC.asm` come from the
folder (`folder_spec` / `memory_spec`), so they always match. Cases and assembly default to
the harness module's containing package; `SPEC.package` can name a different
resource package. No per-kernel `__main__.py` or registration list is needed.


### Execution configuration

Every harness derives from `IpuApp`. Declare its arithmetic and
storage requirements in `SPEC`, independently of interactive debugging:

```python
from ipu_apps.kernel_registry import ExecutionConfig, folder_spec

SPEC = folder_spec(
    MyApp,
    # op, supports, build, and the other kernel fields ...
    execution=ExecutionConfig(mode="fp32"),
)
```

Modes are `native` (encoded elements), `fp32`, and `int32` (four-byte vector
elements). `dtype` defaults to `DType.INT8`; `quantize_output` defaults to
`False`, matching `IpuState()` defaults. The configuration is immutable and
never contains mutable emulator state.

For parameter-dependent arithmetic, use a selector receiving the constructed
harness, for example fully connected's declaration:

```python
execution=lambda app: ExecutionConfig(
    mode="int32" if app.wide_mode else "native", dtype=app.dtype,
)
```

The registry's `create_state(app)` builds fresh state; the base harness calls
it for each run. `create_harness()` binds the selected spec. Directly constructed
harnesses use their class module's matching `SPEC` without a registry scan.
Unregistered harnesses retain native defaults.

An explicit `app.run(state=...)` bypasses state creation and configuration
selection. Existing kernel setup and validation still run. Activation-alpha
and debugger callback behavior remain unchanged.


## Debug a registered kernel

```bash
bazel run --config=debug //src/tools/ipu-apps:identity
bazel run --config=debug //src/tools/ipu-apps:softmax_rows_partial -- --n 32 --rows 8
```

This uses the same registry harness and input cases as normal execution,
opening the TUI after setup and before the first instruction. Kernel code
needs no debugger imports or special entry point. F8 steps, F5 continues, F9
toggles a breakpoint, F10 runs to the selected instruction, and F11 maximizes
the focused pane. `q` or Ctrl-C cancels cleanly without checking partial output.
The terminal must be interactive; the deprecated CLI is not a TUI fallback.

`--config=debug` is a native repository Bazel configuration. Use full targets
for both `bazel run` and `bazel test`; no custom `bazel debug` command or wrapper
is installed. See [debugger documentation](../../../docs/content/debugging.md) for all controls.
