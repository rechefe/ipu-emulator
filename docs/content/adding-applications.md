# Adding applications

How to contribute a kernel so the registry can find it, route to it, and report
its coverage accurately. See [Application coverage](app-coverage.md) for why the
system is shaped this way.

Adding an application is **purely additive**: you create files inside your own
app package and touch no central routing logic.

The smallest complete reference is
`src/tools/ipu-apps/src/ipu_apps/kernels/reshape/identity/`: its assembly
kernel copies an FP32 matrix, while its Python harness only loads the input
memory, configures the run, and reads the output memory.

## What you deliver

| # | Item | Where |
|---|---|---|
| 1 | The `.asm` kernel | `src/ipu_apps/kernels/<family>/<app>/<app>.asm` |
| 2 | An `IpuApp` harness | `src/ipu_apps/kernels/<family>/<app>/app.py` |
| 3 | A `KernelSpec` named `SPEC` | same `app.py`, at the bottom |
| 4 | Runtime cases | Adjacent `cases.py` |
| 5 | Pytest tests | Adjacent `test.py` |

`__init__.py` stays empty — it exists only to mark the directory as a package.
`test.py` is required and starts with one shared line that runs every case in
`cases.py`; add any kernel-specific tests below it:

```python
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__)
```

To also run the default case across sizes, list option overrides as a `sweep`,
rather than writing a parametrized test. Each entry is checked by the case,
and the kernel is assembled once:

```python
test_case = case_tests(__package__, sweep=[
    dict(rows=8, n=200, scale=4.0, seed=1),
    dict(rows=300, n=129, scale=5.0, seed=15),   # crosses a 128-row group
])
```

Optionally, a `benchmark.py` beside them adds `:benchmark_<app>`. It is a
declaration like `cases.py`, not a script: `CONFIGS` lists overrides of the
default case's options, and each runs through that case, checks included.

```python
CONFIGS = [dict(rows=8, n=200), dict(rows=16, n=500)]
PER = "rows"  # optional: adds a cycles-per-row column
```

Kernels of one family keep shared code in the family directory under the same
names: `<family>/app.py` for a shared harness base or spec helpers,
`<family>/cases.py` for shared case builders, and `<family>/test.py` for tests
that span the family (see `kernels/pooling/`, `kernels/convolutions/`,
`kernels/softmax/`). `bazel test :<family>` runs every kernel in the family plus
the family's `test.py`.

Every family has the same shape, `kernels/<family>/<app>/` — e.g.
`kernels/convolutions/conv1x1/` and `kernels/softmax/softmax_rows/`.

No `BUILD.bazel` edit is needed: `ipu_apps_from_kernels` (in `//:ipu_app.bzl`)
globs every `.asm` file under `src/ipu_apps/kernels/` and declares one
`ipu_app` per file automatically, so `bazel run`/`bazel test` labels for a new
kernel appear as soon as its `.asm` is checked in.

## 1. The harness

Subclass `IpuApp`, implement `setup` (load XMEM, set CRs) and `teardown`
(read results back). Declare arithmetic requirements in `SPEC.execution`, for
example `ExecutionConfig(mode="fp32")`; no arithmetic-specific subclass or
state-construction override is needed. The registry creates fresh state for
each run unless the caller supplies one explicitly.

The harness is orchestration, not an implementation of the operation. It must
not calculate expected results or reproduce the kernel in Python. The identity example loads preformatted memory directly. Other kernels may
need layout-specific packing in setup and unpacking in teardown; these hooks
must preserve the declared input/output contract.

**The output file must have the same layout as the input file.** Internal
packing and padding must be reversed by output extraction so that callers do
not need kernel-specific unpacking.

The complete load/run/read flow is:

```python
from ipu_apps.kernel_registry import create_harness

app = create_harness(
    "identity",
    params={"shape": (4, 128)},
    bindings={
        "inst_path": "identity.bin",
        "input_path": "input.bin",  # four preformatted 512-byte XMEM rows
        "output_path": "output.bin",
    },
)
state, cycles = app.run()
```

## 2. The spec

At the bottom of your `app.py`:

```python
def _supports(**params):
    q = softmax_query(params["shape"], params["dim"])
    if bad := positive_dims(q):
        return no(bad)
    if not q.along_rows:
        return no("reduces down columns, not along rows")
    if q.n > LANES:
        return no(f"needs a row of at most {LANES} elements; this row has {q.n}")
    return yes()


def _build(**params):
    q = softmax_query(params["shape"], params["dim"])
    return {"n": q.n, "rows": q.rows}


def _explain(**params):
    q = softmax_query(params["shape"], params["dim"])
    return f"n ({q.n}) < {LANES}: packed row kernel, ..."


SPEC = folder_spec(
    MyKernelApp,               # name = its folder, asm = <folder>.asm
    op="softmax",              # queries route by this first
    variant="rows_partial",    # distinguishes kernels of the same op
    tags=("fp32-wide",),
    requires=("shape", "dim"),  # params the callbacks below index
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **p: (WIDE_VECTOR_ONLY,),
    cost=lambda **p: 1.0,
)
```

### `requires` — the parameters your callbacks index

Callbacks receive `**params` and index it, so a query missing one would raise
`KeyError` from inside `supports`. Naming them here makes the registry check
first and refuse with *"needs parameter 'shape'"* — and keeps a `KeyError`
raised *inside* your callback recognisable as the bug it is, rather than being
downgraded to "no kernel covers this".

### `supports` — what the kernel *can* do

State the kernel's **true domain**, not the range where you would prefer it to
be chosen. If your kernel handles a case that another kernel handles better,
still say yes; `cost` decides the winner.

Getting this wrong in the narrowing direction is a real bug: an early draft of
`softmax_rows_partial` declared `n < 128`, but the kernel genuinely handles
`n == 128` (its `P=1` case), and its own constructor guard — which delegates to
`supports` — then rejected shapes the app had always supported.

Return `no("reason")` with a reason a user can act on. Refusals are aggregated
when nothing covers a query, and they are the only thing the user sees.

### `cost` — which kernel *should* win

Lower wins. Ties are broken by name, so resolution never depends on discovery
order. Use it to express specialisation:

```python
# softmax_rows_partial also handles n == 128, but softmax_rows is the
# specialised full-width path, so step aside at exactly that width.
cost=lambda **p: 2.0 if softmax_query(p["shape"], p["dim"]).n == LANES else 1.0
```

### `caveats` — limits that still apply

Computed per query, so they can quantify the real cost rather than warn
generically:

```python
f"width {width} pads to {padded} elements, so {padded - width} of every "
f"{padded} elements sit idle ({width / padded:.0%} utilisation)."
```

### `bundle` — if you reinterpret the query

If your kernel flattens, pads, or derives shapes, return a `ShapeBundle` so the
verdict discloses it. Never reinterpret a caller's shape silently.

## 3. Delegate your constructor guard

Do **not** restate the domain in `__init__`:

```python
def __init__(self, *, n: int, rows: int, **kwargs) -> None:
    ...
    SPEC.guard(shape=(self.rows, self.n), dim=1)
```

`SPEC.guard` raises `ValueError` with the same reason the registry reports, so
the guard and the router can never drift apart. Hand-written bounds are how the
stale *"use softmax_columns for width >= 128"* message survived a boundary
change.

## 4. Supporting a framework layer

If your op maps to a framework layer, register an adapter **next to your
kernel**:

```python
@register_layer("Conv2d")
def _conv2d(layer, input_shape):
    if layer.groups != 1:
        raise UnsupportedLayer("grouped convolution has no kernel")
    return "conv2d", {...}
```

Put it in your kernel's `app.py` or a module that `app.py` imports (softmax
kernels import the family's `softmax/app.py`). Adapters register as an import
side effect, and discovery imports every kernel's `app.py` — and nothing else —
so `lookup_layer` will find it; nothing central needs editing.

Two obligations:

- **Refuse configuration you do not model.** Ignoring an unrecognised attribute
  makes the registry answer for an operation your kernel does not implement.
- **Refuse look-alike layers explicitly.** `LogSoftmax` shares `Softmax`'s
  signature and computes something else.

## 5. Verify

Your kernel is picked up automatically by the generic checks in
`test/test_kernel_registry.py` — you do not write these:

- **routing** — every case in your `cases.py` must be accepted by your own
  spec and routed by the registry. Cases are what prove your kernel computes
  correctly, so give them an independent reference (e.g. NumPy).
- **spec hygiene** — unique names, the folder / `SPEC.name` / `.asm` agreement,
  well-formed fields.

Softmax kernels additionally get **conformance** (the registry is asked for a
kernel for many shapes, which is then assembled, run and compared against
NumPy) and **guard agreement** (a constructor must not accept what its spec
refuses). These are softmax-only: other operations' kernels each take their own
preformatted input, so there is no shared way to feed them a shape.

Add your own numerical test for the shapes you care about, then:

```bash
bazel test //src/tools/ipu-apps:all
```

## 6. Bazel

Nothing to add here. `ipu_apps_from_kernels` (in `//:ipu_app.bzl`) globs every
`.asm` file under `src/ipu_apps/kernels/` and declares one `ipu_app` label per
file, deriving the kernel's name and source package from the `.asm` path and
its fixture data (sibling `.asm`/`.bin` files) from the same directory. The
same label runs the registry frontend with `bazel run` and the kernel's
`test.py` with `bazel test`. The older `test_<name>` label remains an alias.
The library glob picks up Python files automatically, so as soon as your
kernel's `.asm` is checked in, its case suite is part of CI. See the standard
contract below.

## Checklist

- [ ] Output file layout matches input file layout
- [ ] `SPEC` declares the kernel's true domain, not its preferred range
- [ ] `cost` expresses specialisation; overlaps are intentional
- [ ] Constructor guard delegates to `SPEC.guard`
- [ ] Refusal reasons are actionable
- [ ] Reinterpretation (flatten/pad/derive) is disclosed via `bundle`
- [ ] Layer adapter refuses unmodelled config and look-alike layers
- [ ] `bazel test //src/tools/ipu-apps:all` passes

## Checking your work

```bash
bazel run //src/tools/ipu-apps:query                                  # coverage report
bazel run //src/tools/ipu-apps:query -- softmax shape=32,300 dim=1    # which kernel, and why
bazel run //src/tools/ipu-apps:query -- softmax shape=8,n dim=1 --sweep n=1..200   # routing table
```

`query` is op-agnostic: parameters are `NAME=VALUE` pairs handed to `resolve`
verbatim, so your kernel's op works there as soon as its `SPEC` is discovered.

```python
from ipu_apps.kernel_registry import load, report, resolve

load().skipped        # () -- if your module is here, it failed to import
resolve("softmax", shape=(32, 300), dim=1).describe()
print(report())
```

A kernel that fails to import does not raise; it is recorded as skipped and
reported. If your kernel is missing from coverage, check there first.


## Standard runnable kernel contract

Keep the harness and `SPEC` together in the kernel's `app.py`, with an empty
`__init__.py` and adjacent `cases.py` and `test.py`. A kernel
whose caller supplies a complete FP32 XMEM image should subclass
`kernel_registry.memory.MemoryApp` and declare its SPEC with
`memory_spec(op, App)` — the harness is then just a `memory_layout` classmethod
(see `kernels/reshape/identity/`). The kernel's name and `.asm` come from its
folder, and `requires` from `memory_layout`'s required keyword arguments; the
folder, `SPEC.name`, the `.asm` stem and the Bazel target are one word, and
`test_kernel_registry` checks it.
`kernel_registry/case_support.py` holds the shared case builders
(`tile_input`, `untile_output`, `prepared_image`, `tiled_case`). Cases and assembly
use the harness's containing package (the directory `app.py` lives in), not
`app.py` itself; `SPEC.package` can explicitly select a different resource
package when the class and its spec live in different modules — use the
registry factory for harness construction in that case. Declare reusable
`CASES` in `cases.py` without pytest imports; tests import those cases and use registry
`create_harness(name, params=..., bindings=...)` / `run_case(name, case)` for
construction and execution. The shared runner is used by every `ipu_app`
Bazel declaration; individual kernels need no executable Python entry point.

Run `bazel run //src/tools/ipu-apps:identity -- --list-cases` for an executable
example. `--case` selects an input case, and case options such as `--rows`
override its defaults. `bazel test //src/tools/ipu-apps:identity` exercises
the same harness factory and cases. See the ipu-apps README for the declaration
and case interfaces. Shape-based `resolve()` remains available independently.

Runtime checks must raise descriptive exceptions, including numerical error and
tolerance where applicable. Assertions are reserved for pytest tests. Case option
defaults must be str, int, float, or bool, with names that do not conflict with
shared CLI options. `--output` preserves completed output even when checks fail.

Pass runtime dependencies in `ipu_app(deps=...)` and pytest dependencies in
`test_deps`.
