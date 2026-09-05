# ipu-apps

IPU application test harnesses — Python ports of the C test harnesses.

## Framework

Subclass `IpuApp`, write `setup` and `teardown`, call `run`:

```python
from ipu_apps import IpuApp

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
from ipu_apps.fully_connected import FullyConnectedApp

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
bazel test //src/tools/ipu-apps:test_fully_connected
```


## Registered harnesses and reusable cases

Each kernel has an `__init__.py` (harness hooks and `SPEC`), its assembly,
a `cases.py` (runtime inputs and output checks), and a `test.py` (pytest tests).
Kernel-specific layout and register setup stay in the harness; assembly, construction, execution, and
case checking use the shared registry utilities.

```bash
bazel run //src/tools/ipu-apps:identity
bazel run //src/tools/ipu-apps:softmax_rows_partial -- --rows 8 --n 32
bazel run //src/tools/ipu-apps:identity -- --list-cases
bazel run //src/tools/ipu-apps:identity -- --case single_row
bazel test //src/tools/ipu-apps:test_identity
```

The seven runnable kernels are `fully_connected`, `identity`, `softmax_rows`,
`softmax_rows_partial`, `softmax_rows_long`, `softmax_columns`, and
`softmax_columns_packed`. Each has a `test_<name>` target. Existing softmax
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
the selected kernel name and the same parameters to `create_harness`.
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

Add one declaration in the apps BUILD file:

```starlark
ipu_app(
    name = "my_kernel",
    kernel_package = "src/ipu_apps/my_kernel",
    deps = [":ipu_apps_lib"],
    test_deps = [requirement("pytest")],
    # data = [...],  # Optional input fixtures.
)
```

Use `<name>.asm` in that package and declare the matching `SPEC.name`. For a
different assembly filename, pass `asm="filename.asm"` to the macro and set
`SPEC.asm` to the same path. Cases and assembly default to the harness module's
containing package; `SPEC.package` can name a different resource package.
The macro supplies the shared frontend and creates `test_my_kernel` from the
adjacent `test.py`. No per-kernel `__main__.py` or registration list is needed.


### Execution configuration

Every harness inherits directly from `IpuApp`. Declare its arithmetic and
storage requirements in `SPEC`, independently of interactive debugging:

```python
from ipu_apps.kernel_registry import ExecutionConfig, KernelSpec

SPEC = KernelSpec(
    # name, app_class, supports, build, asm, and the other kernel fields ...
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
If multiple specs declare the same class, construct through `create_harness()`
to select one explicitly. Unregistered harnesses retain native defaults.

An explicit `app.run(state=...)` bypasses state creation and configuration
selection. Existing kernel setup and validation still run. Activation-alpha
and debugger callback behavior remain unchanged.
