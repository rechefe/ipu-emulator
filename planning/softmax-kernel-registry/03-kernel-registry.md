# kernel registry: kernels declare themselves, coverage is generated

### Description

Choosing an app for a computation currently means reading five docstrings and
their constructor guards. This adds a registry that answers the question — and,
more importantly, one that cannot go stale.

The design constraint is stronger than "add a router": **no central file may
contain routing rules, and no coverage claim may be written by hand.** See the
umbrella for why.

```
ipu_apps/kernel_registry/
  shapes.py     role-keyed ShapeBundle, flattening, MalformedQuery
  spec.py       KernelSpec / Verdict / Support
  discovery.py  recursive walk, tolerant of unimportable modules
  registry.py   cost-ordered resolution
  layers.py     framework-layer adapters
  coverage.py   generated reports
```

Three query levels funnel to one resolver:

```python
lookup_layer(nn.Softmax(dim=1), input_shape=(32, 300))   # framework-native
resolve("softmax", shape=(32, 300), dim=1)               # framework-free
lookup(axis="rows", n=300, rows=8)                       # existing API, now a wrapper
```

### Design points worth reviewing

**`supports` says what a kernel CAN do; `cost` says which SHOULD win.**
`softmax_rows_partial` genuinely handles n == 128 (its P=1 case), so it and
`softmax_rows` both claim that width. Declaring its domain as `n < 128` broke
three previously-passing tests, because constructor guards now delegate to
`supports`. Overlap is normal; the specialised kernel wins on cost and the other
is reported as an alternative.

**A malformed question is not missing coverage.** A `dim` outside the input's
rank raises `MalformedQuery` — no kernel could ever answer it — while a shape
nothing handles refuses with every candidate's reason. Folding both into
refusals swallowed genuine typos.

**Guard agreement is checked one-directionally.** Some kernels cannot express
the refused case at all: `softmax_rows` takes only `rows`, its width being fixed
by the `.asm`, so there is no argument on which to reject.

**torch stays optional.** Adapters match on the layer's class name; the registry
never imports torch. Adapters refuse what they do not model — `LogSoftmax` and
`Softmin` share `Softmax`'s signature but compute something else, and
`Softmax(dim=None)` relies on a deprecated torch heuristic.

**Nothing is reinterpreted silently.** A rank > 2 input is flattened and the
reshape is stated in the verdict; shapes the registry derived are marked as
derived; an interior reduction axis is refused rather than transposed.

### What this removes

`softmax_columns_packed` told callers to "use softmax_columns for width >= 128"
long after that boundary had moved to 65. Constructor guards now call
`SPEC.guard()` instead of restating bounds, so the guard and the router cannot
disagree — there is only one statement of the rule.

### Acceptance

- [ ] Routing identical to the previous hand-written router across a full sweep
- [ ] Zero cases where a "supported" verdict fails to construct
- [ ] Generic conformance suite runs whichever kernel the registry resolves and
      compares against numpy — an over-claiming `supports` fails CI
- [ ] A newly registered kernel inherits conformance without writing tests
- [ ] Constructor guards delegate to `SPEC.guard()`
- [ ] `catalog()` and routing tables are probed, not written
- [ ] Bazel targets added; the conformance target globs `**/*.asm` so new
      kernels need no BUILD edit
- [ ] Full test suite green
