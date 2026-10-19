# Parameter domain and a single validation seam (`plan_case`)

Part of [Epic: Kernel framework](00-epic.md) · ID E06-03 · Size S · Depends on: E06-02 · Behaviour change: none

## Goal

Validate a kernel's parameters exactly once, at one seam that turns
`(manifest, params, format)` into a `CasePlan`. Placement, packing, CR
programming, assembly and the runner take the plan and never re-check.

## Background

- One query is validated 3 times on today's `run_case` path:
  `kernel_registry/cases.py:137` (`spec.guard` before assembly),
  `kernel_registry/registry.py:54` (inside `create_harness`), and the
  constructor (`kernel_registry/memory.py:62`; `kernels/softmax/app.py:108`).
  `fully_connected/app.py:167` and `elementwise/residual_add/app.py:99` guard
  in their constructors too. The other harnesses do not guard at all, against
  `docs/content/adding-applications.md` §3.
- A domain is code today (a `supports` callable per kernel, e.g.
  `kernels/projections/app.py:351-357`). Q80 makes it data (`[params]`).
- `resolve` turns any `TypeError` into "does not accept these parameters"
  (`kernel_registry/registry.py:162`), so a bug in `supports` looks like a
  refusal.

## Implementation

Module `ipu_harness/params.py`:

```python
@dataclass(frozen=True)
class CasePlan:
    manifest: KernelManifest
    format: ElementFormat
    params: Mapping[str, int | float | bool | str]   # complete: defaults filled, types coerced
    shapes: Mapping[str, tuple[int, ...]]            # every tensor shape resolved to ints

def plan_case(manifest: KernelManifest, params: Mapping[str, object],
              fmt: ElementFormat) -> CasePlan:
    """The only place a kernel's parameters are validated. Raises ConfigurationError."""
```

Checks, in order:
1. Format: `fmt` is in `[kernel] formats`, and in the case's `formats` if the
   case lists them.
2. Parameters: unknown names, missing required names, type, `min`/`max`,
   `multiple_of`, `choices`.
3. Shapes: every tensor dimension resolves to a positive int. A dimension is an
   int literal, a parameter name, or a small expression over parameters
   (`"k * 4"`, `"ceil(k / 128)"`). The expression grammar is an `ast` whitelist
   of int literals, names, `+ - * // %` and `ceil`/`min`/`max`. Nothing else is
   evaluated.
4. Element scalars: every `float` parameter bound in `[abi]` encodes exactly in
   `fmt` (E03 codec). If it does not, that is an error that names the value, the
   format and the nearest representable values. It is not rounded silently.

`ConfigurationError` is the "configuration error, setup time only" kind from
Q59. It carries the manifest path and the key that failed.

The registry (E06-16) calls the same `plan_case` (with a cheap
`fmt`-independent variant `check_params`) to answer "does this kernel accept
these parameters?". There is no second domain implementation.

## Tests

- [ ] One test per check above, each asserting the message names the key and the value.
- [ ] A spy shows `plan_case` runs once per case run, and that placement, packing and assembly accept only a `CasePlan`.
- [ ] Shape expressions: `ceil(k / 128)` and `rows * 4` evaluate; `__import__`, attribute access and calls other than the whitelist are rejected.
- [ ] `s = 1.5` with `int8` is rejected; `s = 2.0` with `int8` and `e4` is accepted.

## Acceptance Criteria

- [ ] `plan_case` is the single validation seam. No later stage re-validates.
- [ ] No kernel-specific domain code is needed for a kernel whose domain fits `[params]`.
- [ ] `bazel test //...` is green.
