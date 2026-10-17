# `kernel.toml`

Every key a kernel's `kernel.toml` takes, and a family's `registry.toml`.
For a walk-through that builds a kernel step by step, see
[Write a kernel](../how-to/write-a-kernel.md).

A `kernel.toml` declares everything about a kernel except its program: its
parameters and their domain, its input, output and scratch regions, which
register holds each value, its cases, and how outputs are checked. The
harness reads it once (`ipu_harness.manifest.load_manifest`) and every later
step reads that checked view. An error names the file, the line and the key,
for example `kernels/elementwise/scale/kernel.toml:14: [abi] binds CR1, a
hardwired constant`. An unknown key names the closest valid one.

The text below comes from the loader's own documentation, and the table of
keys from the key lists it checks against, so the page changes when the
schema does.

## Schema

--8<-- "generated/kernel-toml/schema.md"

## `registry.toml`

A family declares its variants, kernels that differ from a template kernel
only in parameter values, as entries of `kernels/<family>/registry.toml`. An
entry is its template's `kernel.toml` with the entry's `params` fixed: each
becomes the parameter's only value. A default the template gives stays, so
a query that leaves the parameter out means that default, and an entry fixing
another value refuses it. Without one the entry's value stands in, and a query
that leaves out a parameter its entries fix differently is refused for it. The
entry runs the template's cases with those values, in the template's formats
unless it narrows them. Its Bazel targets are `//kernels/<family>:<name>`,
`:<name>_nightly` and `:assemble_<name>`.

--8<-- "generated/kernel-toml/registry.md"

Bazel reads the entries from the generated `registry.bzl` beside the TOML.
After editing `registry.toml`, rewrite it:

```bash
bazel run //kernels:update_registry_bzl
```

The family's `registry_bzl_test` fails while `registry.bzl` is stale.
