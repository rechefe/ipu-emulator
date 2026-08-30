# Adding applications

How to contribute a kernel so the registry can find it, route to it, and report
its coverage accurately. See [Application coverage](app-coverage.md) for why the
system is shaped this way.

Adding an application is **purely additive**: you create files inside your own
app package and touch no central routing logic.

## What you deliver

| # | Item | Where |
|---|---|---|
| 1 | The `.asm` kernel | `src/ipu_apps/<family>/<app>/<app>.asm` |
| 2 | An `IpuApp` harness | `src/ipu_apps/<family>/<app>/__init__.py` |
| 3 | A `KernelSpec` named `SPEC` | same `__init__.py`, at the bottom |
| 4 | A test | `test/test_<app>.py` |
| 5 | Bazel targets | `src/tools/ipu-apps/BUILD.bazel` |

Nesting depth is free — discovery recurses, so
`convolutions_universal/conv/conv_universal/` works exactly like
`softmax/softmax_rows/`.

## 1. The harness

Subclass `IpuApp`, implement `setup` (load XMEM, set CRs) and `teardown`
(read results back).

**The output file must have the same layout as the input file.** Internal
layouts are yours to choose — pack, pad, chunk, unpack mid-computation — but
what lands on disk must match what the caller supplied. This is enforced by
`test_softmax_layout_roundtrip.py`, which reshapes the raw output with no
app-specific knowledge.

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

Add a `py_pytest_test` target with your `.asm` in `data`. The `ipu_apps_lib`
glob picks up new `.py` files automatically, but **tests are not discovered** —
a test without a target silently never runs in CI.

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
