# Adding applications

How to contribute a kernel so the registry can find it, route to it, and report
its coverage accurately. See [Application coverage](app-coverage.md) for why the
system is shaped this way.

Adding an application is **purely additive**: you create files inside your own
app package and touch no central routing logic.

The smallest complete reference is
`src/tools/ipu-apps/src/ipu_apps/kernel_registry/identity/`: its assembly
kernel copies an FP32 matrix, while its Python harness only loads the input
memory, configures the run, and reads the output memory.

## What you deliver

| # | Item | Where |
|---|---|---|
| 1 | The `.asm` kernel | `src/ipu_apps/<family>/<app>/<app>.asm` |
| 2 | An `IpuApp` harness | `src/ipu_apps/<family>/<app>/__init__.py` |
| 3 | A `KernelSpec` named `SPEC` | same `__init__.py`, at the bottom |
| 4 | Runtime cases and pytest tests | Adjacent `cases.py` and `test.py` |
| 5 | One `ipu_app` declaration | `src/tools/ipu-apps/BUILD.bazel` |

Nesting depth is free — discovery recurses, so
`convolutions_universal/conv/conv_universal/` works exactly like
`softmax/softmax_rows/`.

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

At the bottom of your `__init__.py`:

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


SPEC = KernelSpec(
    name="my_kernel",          # unique; identifies it in verdicts + coverage
    op="softmax",              # queries route by this first
    variant="rows_partial",    # distinguishes kernels of the same op
    app_class=MyKernelApp,
    asm="my_kernel.asm",
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

Put it in a module the kernel package imports (softmax uses
`softmax/_spec_support.py`). Adapters register as an import side effect, and
discovery imports your package, so `lookup_layer` will find it — nothing
central needs editing.

Two obligations:

- **Refuse configuration you do not model.** Ignoring an unrecognised attribute
  makes the registry answer for an operation your kernel does not implement.
- **Refuse look-alike layers explicitly.** `LogSoftmax` shares `Softmax`'s
  signature and computes something else.

## 5. Verify

Your kernel is picked up automatically by the generic suites in
`test/test_kernel_registry.py` — you do not write these:

- **conformance** — the registry is asked for a kernel, then that kernel is
  assembled, run, and compared against NumPy. An over-claiming `supports` fails
  here.
- **guard agreement** — a constructor must not accept what its spec refuses.
- **spec hygiene** — unique names, a resolvable `asm` path, well-formed fields.

Add your own numerical test for the shapes you care about, then:

```bash
bazel test //src/tools/ipu-apps:all
```

## 6. Bazel

Add one `ipu_app` declaration with the kernel's name, source package, and
optional fixture data. It creates both the run target and `test_<name>` using
the shared registry frontend and the adjacent `test.py`. The library glob
picks up Python files automatically; the Bazel declaration makes the kernel's
case suite part of CI. See the standard contract below.

## Checklist

- [ ] Output file layout matches input file layout
- [ ] `SPEC` declares the kernel's true domain, not its preferred range
- [ ] `cost` expresses specialisation; overlaps are intentional
- [ ] Constructor guard delegates to `SPEC.guard`
- [ ] Refusal reasons are actionable
- [ ] Reinterpretation (flatten/pad/derive) is disclosed via `bundle`
- [ ] Layer adapter refuses unmodelled config and look-alike layers
- [ ] Bazel target added
- [ ] `bazel test //src/tools/ipu-apps:all` passes

## Checking your work

```bash
python -m ipu_apps.softmax --catalog          # coverage table
python -m ipu_apps.softmax --shape 32,300 --dim 1
```

```python
from ipu_apps.kernel_registry import load, report, resolve

load().skipped        # () -- if your module is here, it failed to import
resolve("softmax", shape=(32, 300), dim=1).describe()
print(report())
```

A kernel that fails to import does not raise; it is recorded as skipped and
reported. If your kernel is missing from coverage, check there first.


## Standard runnable kernel contract

Keep the harness and `SPEC` in the kernel's `__init__.py`, with assembly and
adjacent `cases.py` and `test.py` modules. Declare reusable `CASES` in `cases.py`
without pytest imports; tests import those cases and use registry
`create_harness(name, params=..., bindings=...)` / `run_case(name, case)` for
construction and execution. The shared runner is used by every `ipu_app`
Bazel declaration; individual kernels need no executable Python entry point.

Run `bazel run //src/tools/ipu-apps:identity -- --list-cases` for an executable
example. `--case` selects an input case, and case options such as `--rows`
override its defaults. `bazel test //src/tools/ipu-apps:test_identity` exercises
the same harness factory and cases. See the ipu-apps README for the declaration
and case interfaces. Shape-based `resolve()` remains available independently.

Runtime checks must raise descriptive exceptions, including numerical error and
tolerance where applicable. Assertions are reserved for pytest tests. Case option
defaults must be str, int, float, or bool, with names that do not conflict with
shared CLI options. `--output` preserves completed output even when checks fail.

Pass runtime dependencies in `ipu_app(deps=...)` and pytest dependencies in
`test_deps`. If assembly is not named `<name>.asm`, set the macro's `asm` and
`SPEC.asm` to the same relative path. Cases and assembly use the harness's
containing package, including for a class in `app.py`; `SPEC.package` can
explicitly select a different resource package. Use the registry factory for
harness construction when the class and its spec live in different modules.
