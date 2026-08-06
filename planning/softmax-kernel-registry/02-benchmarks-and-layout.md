# softmax: benchmarks and a layout contract test

### Description

Two additions on top of the kernels, both about keeping them honest rather than
changing what they compute.

**Benchmarks.** A shared table renderer plus a benchmark package per kernel,
each with its own `CONFIGS` and a committed `results.md`:

```
python -m ipu_apps.softmax.<app>.benchmark.benchmark
```

Every config checks correctness against numpy as well as reporting cycles and
MULT/ACC utilisation, so a benchmark cannot report a fast wrong answer.

Configs deliberately include shapes that cross a 128-row group boundary, where
the kernel re-runs all four passes on the next group. That keeps the per-group
overhead visible in cyc/row instead of hiding behind small inputs — and it
shows the cost is nil: `softmax_rows_partial` at n=128 runs 28.23 cyc/row over
300 rows versus 28.48 over 50.

**Layout contract test.** Asserts that every kernel writes its output in the
same layout as its input: same file size, and a naive reshape of the raw bytes
matches the numpy reference.

### Why the layout test cannot live in the per-app tests

The per-app tests un-pack the output using app-specific knowledge before
comparing. That makes them structurally blind to a regression that merely
**misplaces** elements rather than computing them wrongly — which is exactly
how a packed-output discrepancy went unnoticed until it was found by hand.

The layout test reshapes the raw output file with **no** app-specific
un-packing knowledge, so it catches what the others cannot. Ten cases cover all
five kernels in both a padding-heavy and a zero-padding shape each.

### Acceptance

- [ ] 28 benchmark configs across five kernels, all passing
- [ ] Each config asserts correctness, not only cycles
- [ ] Configs include shapes crossing a group boundary
- [ ] Layout test covers all five kernels, padding-heavy and zero-padding
- [ ] Layout test uses no app-specific un-packing
- [ ] Bazel target added
- [ ] Full test suite green
