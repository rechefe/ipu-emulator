# Kernel coverage

The kernel registry (`ipu_harness.registry`) answers which kernel computes a
given operation with given parameters, and why. It answers from each kernel's
own `kernel.toml`, so no central file holds routing rules: adding a kernel
adds its folder or its `registry.toml` entry and changes nothing else. The
tables at the end of this page are generated from the same registry.

## Why the kernels declare their own domain

A rule written in two places drifts. In the kernels' first form, each kernel's
constructor checked its own bounds and a central router restated them, and the
two disagreed: `softmax_columns_packed` advised callers to use
`softmax_columns` for a width of 128 and above long after that boundary had
moved to 65. Nothing read the message, so nothing failed.

A kernel's `[params]` table is the one statement of its domain: each
parameter's type, bounds, multiple and choices. The same check
(`ipu_harness.params.check_params`) decides whether a case can run
(`plan_case`) and whether routing accepts a query, so the two cannot disagree.

## How a kernel declares what it computes

A kernel declares its operation, formats and routing cost in `[kernel]`, and
its domain in `[params]`
([`kernel.toml` reference](../reference/kernel-toml.md)). From
`kernels/softmax/softmax_rows_partial/kernel.toml`:

```toml
[kernel]
op = "softmax"
formats = ["fp32"]
cost = "1 + min(1, max(0, n - 127))"

[params]
rows  = { type = "int", min = 1 }
n     = { type = "int", min = 1, max = 128 }
dim   = { type = "int", choices = [1], default = 1 }
```

The kernel takes rows of up to 128 elements. At `n = 128`, where
`softmax_rows` also accepts and is cheaper, its cost rises to 2.

A **variant** is a kernel that differs from a template kernel only in
parameter values. A family declares its variants as entries of
`kernels/<family>/registry.toml`:

```toml
[[kernel]]
name = "maxpool2d_nms7"
template = "maxpool2d_nms"
params = { kernel_size = 7, padding = 3 }
doc = "SuperPoint simple_nms, nms_radius = 3"
```

An entry is its template's `kernel.toml` with each of the entry's `params`
fixed to one value. The template itself runs its own cases but is not routed:
its open domain would tie with each of its entries.

## How the registry loads

`Registry.load` reads every `kernels/<family>/<name>/kernel.toml` and every
`kernels/<family>/registry.toml`. It parses TOML only and imports no kernel
Python, so a query never runs kernel code.

A file that fails to load goes into `Registry.invalid` with its error, and the
other kernels still load. The coverage report lists those files: a kernel that
failed to load is a kernel missing from coverage, and a registry that stopped
at an unrelated half-written kernel would fail when it is most needed. Two
kernels with the same name are an error, because a name is one kernel.

## How a query resolves

`Registry.resolve(op, **params)` holds no knowledge of any operation. It asks
each kernel of the operation whether it accepts the parameters:

1. Take the kernels whose `[kernel] op` is `op`.
2. Refuse a kernel when the query leaves out a parameter it marks
   `query_required`.
3. Check the parameters against the kernel's domain with `check_params`, as
   `plan_case` does.
4. With `format=`, refuse a kernel that does not take the parameters in that
   format. Without it, refuse a kernel none of whose formats encodes the
   parameters' element scalars.
5. Among the kernels that accept, choose the lowest `[kernel] cost`. A cost is
   a number or an expression over the integer parameters and derived values.

The verdict names the chosen kernel, its parameters with defaults filled, the
formats it runs them in, its Bazel label and the other kernels that accept,
cheapest first. When no kernel accepts, the verdict's reason names every
kernel's refusal, so the caller learns what was wrong.

Overlapping domains are normal: `softmax_rows_partial` and `softmax_rows`
both accept `n = 128`, and cost settles which one runs. Two accepting kernels
at the same lowest cost are a `ConfigurationError`, never settled by name or
discovery order, so the answer cannot change because a folder was renamed. A
tie between entries that fix a parameter the query leaves out is a refusal
that names the parameter, since the query is incomplete rather than the
registry wrong.

## Framework layers

`Registry.from_layer(layer, input_shape)` takes a framework layer, for example
`nn.Softmax(dim=1)`, and the shape it will receive. A layer carries
configuration (`dim`, `in_channels`, `stride`) but no shape, and the shape
selects the kernel, so both are needed.

A family's adapters are `LAYER_ADAPTERS = {"Conv2d": fn, ...}` in
`ipu_kernels.<family>.layers`, each mapping `(layer, input_shape)` to an
operation and its parameters. The registry imports a family's adapters by
module name the first time `from_layer` needs one, and matches a layer by its
class name, so the registry never imports torch. Adapters follow two rules,
because both failures give confident wrong answers:

- **Translate what you understand; refuse the rest.** An adapter that ignores
  an attribute it does not model answers for an operation the kernel does not
  implement.
- **Never assume a neighbouring layer is equivalent.** `LogSoftmax` and
  `Softmin` sit beside `Softmax` in `torch.nn` with the same signature; the
  softmax adapters refuse them by name.

## Ask the registry from the command line

```bash
bazel run //kernels:query                                         # the coverage report
bazel run //kernels:query -- softmax rows=8 n=300 dim=1           # which kernel, and why
bazel run //kernels:query -- softmax rows=8 n=n dim=1 --sweep n=1..139   # where the winner changes
```

`--sweep` runs `ipu_harness.query.boundaries`: it resolves the query at each
value and collapses runs of the same winner, which is how a family page's
routing table is produced. The table cannot describe behaviour the kernels do
not have, because it is probed. The [softmax page](../kernels/softmax.md)
shows the softmax table. The options are in the
[command-line reference](../reference/cli.md).

## What keeps the registry honest

A declarative registry can fail in one new way: a kernel's domain accepts a
query the kernel computes wrongly. These tests guard it:

- **Cases.** Every kernel's cases run it and compare its output with an
  independent reference.
- **Routing tests.** A family's `test_routing.py` probes the registry at each
  boundary its docs page's routing tables claim, including that the winner
  changes where the table says, and checks that an accepted query also plans
  and places. A domain that moves without the page fails there.
- **Conformance** (softmax, `kernels/softmax/test_conformance.py`). For each
  shape, the registry chooses a kernel, and the test runs it against a NumPy
  reference, so a kernel cannot claim a domain it mishandles.
- **Spec hygiene.** Every `kernel.toml` loads, kernel names are unique, and
  each family's generated `registry.bzl` matches its `registry.toml`.

## Current coverage

Generated from `default_registry()` when the docs are built; it is the report
`bazel run //kernels:query` prints.

--8<-- "generated/kernel-coverage/kernels.md"

### Kernels still in `ipu_apps`

The kernels not yet moved to `kernels/` answer
`bazel run //src/tools/ipu-apps:query` through the `ipu_apps.kernel_registry`
registry, where each kernel's `app.py` declares a `SPEC` with a `supports`
function in place of `[params]`. Generated from that registry:

--8<-- "generated/kernel-coverage/ipu-apps.md"

To add a kernel, see [Write a kernel](../how-to/write-a-kernel.md).
