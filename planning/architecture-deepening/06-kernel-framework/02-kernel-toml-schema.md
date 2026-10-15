# `kernel.toml` schema, loader and read-only typed view

Part of [Epic: Kernel framework](00-epic.md) · ID E06-02 · Size M · Depends on: E06-01, E03 (element-format enum and names) · Behaviour change: none

## Goal

Make one `kernel.toml` file the whole declaration of a kernel, apart from its
program. One loader reads it, checks it against a strict schema, and returns a
frozen typed view. Errors point at `kernel.toml:<line>`. Every later stage reads
the view and never the raw TOML. This follows the pattern Q30 sets for
`instruction_spec.py`.

## Background

- A kernel's declaration is spread over 4 Python files today. Its SPEC is in
  `app.py` (`kernels/projections/app.py:336-383` `projection_spec`), its
  domain is a `supports` callable, its cases are in `cases.py`, its CRs are in
  `setup()` (`kernels/projections/app.py:213-233`) and its file layout is in
  docstrings (`kernels/projections/app.py:44-55`).
- Q80 fixes the sections: `[kernel]` (op, formats), `[params]` (domain),
  `[inputs]`/`[outputs]` (shapes), `[abi]` (name -> CR), `[cases]` (including
  sweeps and user files), and `[golden]` (reference selection and tolerance
  tier per format).

## Implementation

Module `ipu_harness/manifest.py`. TOML is parsed with `tomllib` (stdlib), and
line numbers are recovered with a small key locator over the source text.

```python
@dataclass(frozen=True)
class ParamDecl:
    name: str
    type: Literal["int", "float", "bool", "str"]
    min: int | float | None = None
    max: int | float | None = None
    multiple_of: int | None = None
    choices: tuple[object, ...] | None = None
    default: object | None = None
    unit: Literal["count", "rows", "bytes"] = "count"   # L16: bytes only for listed legacy kernels

@dataclass(frozen=True)
class TensorDecl:
    name: str
    shape: tuple[int | str, ...]       # ints or param names / simple expressions (E06-03)
    row: int | None = None             # Q81 pin
    pack: str | None = None            # named transform (E06-05)
    store: Literal["element", "accumulator"] = "element"   # outputs only
    role: Literal["input", "output", "scratch"]

@dataclass(frozen=True)
class CaseDecl:
    name: str
    values: Mapping[str, object]       # scalars or lists (lists = sweep axes)
    formats: tuple[ElementFormat, ...] | None
    inputs: Mapping[str, Path]         # user .npy (Q82)
    expected: Mapping[str, Path]
    generator: str | None              # "module:fn" escape hatch
    tier: Literal["default", "nightly"] = "default"   # Q73

@dataclass(frozen=True)
class GoldenDecl:
    reference: str | None              # "reference.py:fn"; default "reference.py:reference" if present
    op: str | None                     # shared reference by op (ipu_kernels.reference)
    tolerance: Mapping[ElementFormat, str]   # tier name per format (E06-10)

@dataclass(frozen=True)
class KernelManifest:
    name: str                          # folder name, or the registry.toml entry name (E06-17)
    directory: Path
    asm: Path
    op: str
    formats: tuple[ElementFormat, ...]
    params: Mapping[str, ParamDecl]
    tensors: Mapping[str, TensorDecl]
    abi: Mapping[str, CrIdx]           # validated "CR2".."CR15"
    derived: Mapping[str, str]         # see E06-06 open question
    cases: Mapping[str, CaseDecl]      # must contain "default"
    golden: GoldenDecl
    stages: str | None                 # "stages.py:run" (E06-14)

def load_manifest(directory: Path, *, overrides: Mapping | None = None) -> KernelManifest: ...
```

Schema rules, checked at load:

- Tensor sections are `[inputs.<name>]`, `[outputs.<name>]` and
  `[scratch.<name>]`. A scratch region is placed and zero-filled, or filled
  with `fill = <value>` (for example the log2(e) row that the softmax harnesses
  write today, `test/test_alias_profile_kernels.py:38-39`). It is never
  compared.
- Unknown section or key is an error, with the closest valid key suggested.
  Element format names are the E03 names (`fp32`, `int8`, `e1`..`e7`, `int32`).
- `[abi]` values are `CR2`..`CR15`. `CR0` and `CR1` are hardwired constants, so
  binding them is an error. `CR15` is allowed (CLAUDE.md: any CR is valid). If a
  kernel binds `CR15`, the framework does not write the default dstructure
  (E06-06).
- Every `[abi]` key names a parameter, a tensor or a `derived` value. Every
  tensor has an `[abi]` entry, because its base row must reach the program.
- `[cases.default]` exists. Case keys are parameter names or the reserved keys
  `formats`, `inputs`, `expected`, `generator`, `tier`.
- File paths (`inputs`, `expected`, `reference`, `generator`, `stages`) stay
  inside the kernel directory. `..` and absolute paths are rejected.
- `overrides` is how a family `registry.toml` entry (E06-17) supplies parameter
  defaults and a name over a template's `kernel.toml`.

The view is immutable: `MappingProxyType` and frozen dataclasses. Construction
is the only place that touches raw TOML.

## Tests

- [ ] A valid manifest (the `scale` example from the epic) loads into the expected view.
- [ ] Each rule above has a failing fixture whose error names the file, the line and the key (`kernel.toml:14: [abi] binds CR1, a hardwired constant`).
- [ ] An unknown key suggests the nearest valid key.
- [ ] Mutating the view raises.
- [ ] Format names are validated against the E03 enum, with no local copy of the list.

## Acceptance Criteria

- [ ] `load_manifest` is the only reader of `kernel.toml` in the repo.
- [ ] The schema is documented once, in the module docstring, which feeds the E06-21 reference page.
- [ ] `bazel test //...` is green.
