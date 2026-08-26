# Application coverage

This page describes how the emulator knows **which application implements a
given computation**, and how that knowledge is kept honest as applications are
added.

The mechanism is the *kernel registry* (`ipu_apps.kernel_registry`). Its design
goal is narrow and worth stating up front: **no central file may contain
routing rules.** Adding an application must be a purely additive change.

## The problem it solves

Before the registry, choosing an app meant reading five docstrings and their
constructor guards, and a central router restated every app's bounds:

```python
# the app's own guard
if self.width < MIN_WIDTH: raise ValueError(...)

# ...and the router, restating the same rule
if width <= 64: → packed
else:           → columns
```

A rule written twice drifts. It did: `softmax_columns_packed` advised callers
to *"use softmax_columns for width >= 128"* long after that boundary had moved
to 65. Nothing broke, because nothing read the message — which is exactly how
coverage documentation rots without anyone noticing.

The registry removes the second copy.

## How a kernel declares itself

Each app package exports a module-level `SPEC` beside its `.asm` and harness:

```python
SPEC = KernelSpec(
    name="softmax_columns",
    op="softmax",
    variant="columns",
    app_class=SoftmaxColumnsApp,
    asm="softmax_columns.asm",
    requires=("shape", "dim"),  # params the callbacks index
    supports=_supports,   # params -> Support (the domain)
    build=_build,         # params -> constructor kwargs
    explain=_explain,     # params -> why this kernel
    caveats=_caveats,     # params -> limits that still apply
    cost=lambda **p: 1.0, # tie-break against other claimants
)
```

`supports` is **the single source of truth for the kernel's domain**. The app's
own constructor delegates to it:

```python
def __init__(self, *, rows: int, width: int, **kwargs) -> None:
    ...
    SPEC.guard(shape=(self.rows, self.width), dim=0)
```

so the guard and the router cannot disagree.

## Discovery

`discover()` walks the `ipu_apps` package tree and collects every `SPEC`. Two
properties matter:

- **Arbitrary depth.** Softmax nests kernels one level down; the convolution
  family nests them three (`convolutions_universal/conv/conv_universal/`).
- **Tolerance of broken packages.** A working tree routinely contains modules
  that will not import — a kernel mid-authoring, a stale directory left by a
  branch switch, an uninstalled optional dependency. Those are recorded as
  `SkippedModule` and reported, never raised. A registry that dies on an
  unrelated half-finished app would fail exactly when it is most needed.

Skipped modules appear in the coverage report, because a module that failed to
import is a kernel potentially missing from coverage.

## Resolution

The registry holds **no per-operation knowledge**. It asks every kernel
registered for the operation whether it handles the query, and picks the
cheapest one that says yes:

1. filter kernels by `op`;
2. call each kernel's `supports(**params)`;
3. among those that accept, choose the lowest `cost`;
4. report the rest as `alternatives`.

Ordering by declared `cost` — never by discovery order — keeps resolution
deterministic. An order-dependent registry is the classic way this kind of
system becomes quietly non-reproducible: the answer changes because a file was
renamed.

Overlap is normal and expected. `softmax_rows_partial` genuinely handles
`n == 128` (that is just its `P=1` case), so both it and `softmax_rows` claim
that width. `supports` states what a kernel **can** do; `cost` decides which
**should** win.

When nothing accepts, the refusal aggregates every kernel's reason, so the
caller learns what was wrong rather than only that it failed.

## Shapes travel as a bundle

A query is never "one shape". A softmax has an input and an output; a matmul
has two independent inputs; a convolution has input, weight, bias and output.
Shapes therefore travel as a role-keyed `ShapeBundle`
(`input`/`weight`/`bias`/`output`, or an op's own roles).

Two properties of the bundle exist to prevent silent misreporting:

- **Provenance.** Shapes the registry *derived* (an output shape computed from
  a layer's parameters) are recorded in `derived_roles` and rendered with a
  `*`. A wrong derivation must never read as something the caller asserted.
- **Disclosed reinterpretation.** When a rank > 2 input is flattened, the note
  goes on the bundle and into the verdict. A reshape the caller did not ask for
  is never silent.

Flattening around an *interior* axis is refused rather than performed, since it
would require transposing the other dimensions — silently reinterpreting the
caller's memory layout.

## Framework layers

`lookup_layer(layer, input_shape)` accepts a framework layer directly. A layer
carries *configuration* (`dim`, `in_channels`, `stride`) but never *shape* —
`nn.Softmax(dim=1)` has no idea it will receive a 32×300 — and shape is what
selects a kernel, so both are required.

Adapters are matched on the layer's **class name**, so the registry never
imports torch and torch stays an optional dependency.

Every adapter lives beside the kernels it serves — softmax's are in
`ipu_apps/softmax/_spec_support.py`, not in the registry core — so no
operation's vocabulary leaks into the op-agnostic layer. They register as an
import side effect of that package, which is why `from_layer` runs discovery
before looking one up: an adapter in a package nothing has imported yet has not
registered.

Adapters follow two rules, because both failure modes are silent:

- **Enumerate what you understand; refuse the rest.** An adapter that ignores
  an attribute it does not model will answer for an operation the kernel does
  not implement.
- **Never assume a neighbour is equivalent.** `LogSoftmax` and `Softmin` sit
  beside `Softmax` in `torch.nn` and share its signature; routing them to a
  softmax kernel would return confidently wrong numbers. They are refused by
  name.

## Coverage reports are generated

`report()` and `boundaries()` obtain every figure by **asking the kernels**.
`boundaries()` sweeps a parameter and collapses runs of the same winner, which
is how routing tables are produced:

```python
boundaries("softmax", "shape", range(1, 140),
           build=lambda n: (8, n), dim=1)
```

```
n (row length) 1..127        softmax_rows_partial
n (row length) = 128         softmax_rows
n (row length) 129..139      softmax_rows_long
```

Because the table is probed rather than maintained, it cannot describe
behaviour the kernels do not have.

The routing tables printed in these pages *are* checked-in text, though, so
`test/test_docs_routing.py` reads them back and probes the registry at every
boundary they claim — including that each run really does change hands where it
says. A kernel domain that moves without the docs following fails there.

## What keeps it honest

A declarative registry can fail in a new way: `supports` says yes and the
kernel disagrees. Two mechanisms guard that, both in
`test/test_kernel_registry.py` and both generic — a newly registered kernel
inherits them automatically:

- **Conformance.** For each shape, the registry is asked for a kernel, and that
  kernel is then assembled, run, and compared against a NumPy reference. A
  kernel cannot claim a domain it mishandles.
- **Guard agreement.** A constructor must not accept a shape its spec refuses.
  (Checked one-directionally: some kernels cannot express the refused case at
  all — `softmax_rows` takes only `rows`, its width being fixed by the `.asm` —
  so there is no argument on which to reject.)

## Current coverage

Generated by `report()`; see the routing tables above for the boundaries.

| operation | kernels |
|---|---|
| `softmax` | `softmax_rows`, `softmax_rows_partial`, `softmax_rows_long`, `softmax_columns`, `softmax_columns_packed` |
| `conv2d` | `conv_universal`, `conv_universal_bn_activation`, `conv_first_layer`, `conv_universal_wide384`, `depthwise_conv_universal`, `depthwise_conv_universal_bn_activation`, `depthwise_conv_stride2_128`, `depthwise_conv_stride2_narrow`, `depthwise_conv_stride2_16`, `pointwise_conv_unified`, `pointwise_conv_unified_bn_activation` |
| `residual_add` | `residual_add` |

All softmax and conv2d/residual_add kernels are wide-vector FP32 only
(`wide_vector_debug=True`) and have no narrow (INT8/FP8) variant. This is
reported as a caveat on every verdict from either family. See
[Softmax](kernels/softmax.md) and [Convolutions](kernels/convolutions.md) for
per-kernel routing tables and cost.

To add an application, see [Adding applications](adding-applications.md).
