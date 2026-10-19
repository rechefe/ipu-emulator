# Routing registry as a value built from `kernel.toml` files

Part of [Epic: Kernel framework](00-epic.md) · ID E06-16 · Size M · Depends on: E06-03 · Behaviour change: none

## Goal

Answer "which kernel computes this op with these parameters, and why?" from the
`kernel.toml` files, through an explicit `Registry` value. This removes the
import-time global tables and import-the-whole-tree discovery. The routing rules
stay the same: every kernel for the op is asked, and the cheapest one that
accepts wins, never the first in discovery order.

## Background

- The routing rule is documented in `kernel_registry/registry.py:1-12`.
- Discovery imports every `app.py` and memoises per package in a module global
  (`registry.py:28-29` `_LOCK`/`_CACHE`, `registry.py:104-113` `load`). Layer
  adapters live in another global (`layers.py:42` `_ADAPTERS`), filled as a
  side effect of importing `convolutions/app.py` and `softmax/app.py`.
- Tests clean up the globals by hand: `_ADAPTERS.pop`, temp packages with
  `sys.path.insert`, and deleting from `sys.modules` (`test/test_kernel_registry.py`).
- `kernel_spec` reports "unknown kernel" even when the kernel's `app.py` failed
  to import (`registry.py:32-37`).
- `resolve` turns any `TypeError` raised in `supports` into a refusal
  (`registry.py:162`).
- User-facing tools: `bazel run //src/tools/ipu-apps:query` (routing and the
  coverage report, `docs/content/adding-applications.md:281-283`,
  `docs/content/app-coverage.md`) and `:kernel_manifest`.

## Implementation

`ipu_harness/registry.py`:

```python
@dataclass(frozen=True)
class Registry:
    manifests: Mapping[str, KernelManifest]          # by kernel name (folders + registry.toml entries)
    invalid: Mapping[Path, ConfigurationError]       # manifests that failed to load, by path
    adapters: Mapping[str, LayerAdapter]             # collected, not registered by side effect

    @classmethod
    def load(cls, root: Path, *, families: Iterable[str] | None = None) -> "Registry": ...
    @classmethod
    def of(cls, manifests: Iterable[KernelManifest], adapters: Mapping = {}) -> "Registry":  # test seam

    def kernel(self, name: str) -> KernelManifest     # raises the stored load error if that kernel's toml was invalid
    def resolve(self, op: str, **params) -> Verdict  # same Verdict shape as today
    def from_layer(self, layer, shape) -> Verdict

@functools.cache
def default_registry() -> Registry: ...              # one cached value, built from the runfiles kernels/ tree
```

- **Acceptance** is `check_params` from E06-03, the same code `plan_case`
  uses, so there is no second domain implementation. A kernel accepts a query
  if its parameters validate for at least one of its formats, or for the format
  the query names.
- **Cost** is `[kernel] cost`, a constant or a whitelist expression over
  parameters (E06-03 grammar). The default is 0. Ties between accepting
  kernels are an error at load, as today's "decided by cost, never by
  discovery order".
- **Layer adapters** are declared in `ipu_kernels/<family>/layers.py` as
  `LAYER_ADAPTERS = {"Conv2d": fn, ...}`. `Registry.load` collects them. There is
  no decorator side effect.
- **Loading** parses TOML only. No kernel Python is imported to route a query.
  Scoped loading (`families=`) serves per-kernel tests (E06-18).
- **Tools** move to `//kernels:query` and `//kernels:manifest`. Their output
  format is unchanged.

## Tests

- [ ] `Registry.of(...)` tests replace every `_ADAPTERS.pop`, `sys.path` and `sys.modules` manipulation in the registry tests.
- [ ] `resolve` picks the cheapest accepting kernel, and the result does not depend on manifest order (shuffled input).
- [ ] An invalid `kernel.toml` named by `kernel(name)` raises its own load error, not "unknown kernel".
- [ ] A query with an unknown parameter is a refusal that names the parameter. A crash inside a cost expression propagates.
- [ ] `default_registry()` loads without importing any module under `ipu_kernels`, except the `layers.py` files when `from_layer` is used.

## Acceptance Criteria

- [ ] No module-level mutable registry state in `ipu_harness`.
- [ ] `//kernels:query` gives the same routing answers as `//src/tools/ipu-apps:query` for every migrated kernel (checked per family in E07).
- [ ] `bazel test //...` is green.
