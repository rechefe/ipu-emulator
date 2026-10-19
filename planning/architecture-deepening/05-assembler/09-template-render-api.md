# Template render API: declared params, defines, sandboxed macro loader

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-09 · Size M · Depends on: E05-05 · Behaviour change: none (a source without defines or includes renders exactly as before)

## Goal

Make `ipu_as.template.render` the one deep render function every caller uses (L10):

- template values come in as `defines`, checked against declared parameters;
- `{% include %}` and `{% import %}` resolve only inside explicit search roots, through a sandboxed loader;
- diagnostics name the included file and line.

This issue adds the API with today's `Undefined` behaviour. E05-11 flips to `StrictUndefined`, E05-12 moves every caller onto this path, and E05-13 adds the `cr` / `lr` namespaces.

## Background

- `template.render(text)` (`ipu_as/template.py:44-49`) is `SandboxedEnvironment().from_string(text).render()`. It takes no context and has no loader, so there is no include or import.
- Undeclared parameters hide behind `| default(...)`, for example `conv_universal_wide384.asm` `cpr|default(3)` and `rows|default(8)`, or `depthwise_conv_stride2_narrow.asm:3` `elements_in_row|default(32)`. A harness that forgets a key silently assembles the default shape.
- Jinja macros are copied between mobilevit files because there is no import.
- `diagnostics._instrument` (`diagnostics.py:81-130`) marks lines of a single string. It knows nothing about included templates.

## Implementation

```python
# ipu_as/template.py
REQUIRED = object()

@dataclass(frozen=True)
class Param:
    name: str
    type: type                         # int | bool | str | tuple[int, ...]
    default: object = REQUIRED
    choices: tuple | None = None

@dataclass(frozen=True)
class ParamSpec:
    params: tuple[Param, ...]
    def bind(self, defines: Mapping[str, object]) -> dict[str, object]   # check, coerce, fill defaults

def declared_params(text: str) -> ParamSpec | None    # parses an `ipu:params` header comment, None if absent

def render(text: str, *, defines: Mapping[str, object] = {}, params: ParamSpec | None = None,
           search_path: Sequence[Path] = (), source_name: str = "<asm>",
           namespaces: Mapping[str, Mapping[str, str]] = {}) -> str
```

1. **Parameter binding.**
   - `params` is the declared parameter set. When it is `None`, `declared_params(text)` is used, and when that is also `None`, the source declares nothing.
   - With a declaration:
     - every key in `defines` must be declared, else `TemplateError("undeclared define 'cpr'")`;
     - every `REQUIRED` param must be supplied;
     - defaults fill the rest;
     - values are coerced from CLI strings (`"3"` → `3`, `"true"` → `True`, `"1,2,3"` → `(1, 2, 3)`) and checked against `choices`.
   - Without a declaration, a non-empty `defines` is an error. This keeps the "declared params" rule strict.
2. **Header format** for standalone `.asm` files:

   ```jinja
   {#- ipu:params
   cpr:        int = 3            # column chunks per row
   rows:       int = 8
   activation: str = "identity" in (identity, relu, silu)
   -#}
   ```

   The header is a Jinja comment, so it renders to nothing. The parser is about 40 lines, and its error messages carry the header line number.
3. **Loader.** Use a `jinja2.FileSystemLoader(search_path)` wrapped so that:
   - absolute names and names containing `..` are rejected;
   - resolved paths must stay inside a root (checked with `Path.resolve().is_relative_to`);
   - the environment is still a `SandboxedEnvironment`.

   One environment is cached per `search_path` tuple. Library templates use the `*.asm.j2` suffix so kernel discovery (`ipu_app.bzl _kernel_asms`) never mistakes them for kernels.
4. **Errors.** Every jinja failure is wrapped in `TemplateError` (E05-04), with `SourceLocation(source=<template name>, line=…)` and `__cause__` kept. That includes `TemplateSyntaxError`, `UndefinedError`, `SecurityError`, `TemplateNotFound` and runtime errors from expressions.
5. **Diagnostics across includes.** The loader's `get_source` applies the same `#@ipu-src-line:` instrumentation, tagged with the template name (`#@ipu-src:<name>:<line>`). `diagnostics` then reports `matmul_chunk.asm.j2:41` for an error inside a macro. When the error is in an included file, the diagnostic's position in the open file is the `{% include %}` line, and the message names the included file and line.
6. `AssemblySession(source, defines=…, params=…, search_path=…, name=…)` forwards all of these. Remove the "defines not supported yet" guard from E05-05.

## Tests

- [ ] Each of these raises `TemplateError` with a location:
  - an undeclared define;
  - a missing required param;
  - a wrong type;
  - a value outside `choices`;
  - `defines` given to a source with no declaration;
  - an include outside the search path (`../x`, an absolute path, a symlink escaping the root);
  - a `__class__` escape inside an included file.
- [ ] A header with defaults renders identically with `defines={}` and with the defaults passed explicitly.
- [ ] An include from a search root works. `{% import "m.asm.j2" as m %}{{ m.f() }}` works.
- [ ] A diagnostic for an error inside an included macro names the macro file and line.
- [ ] A source without markers is not rendered at all (fast path kept). E02 digests unchanged.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `render` is the only function in `ipu_as` that constructs a Jinja environment.
- [ ] The sandbox is kept with a loader. Path-traversal tests pass.

## Open question

- **Where params are declared.** Q80 puts a `[params]` domain in `kernel.toml` (E06), and L10 asks for "declared params". This issue lets the caller pass a `ParamSpec`, which is how the E06 harness hands over `kernel.toml [params]`. It also reads an in-template `ipu:params` header for standalone `.asm` files (CLI, Bazel `assemble_asm`, tests). Undecided: should the header stay once every registered kernel has a `kernel.toml`, or should `kernel.toml` be the only declaration, with the CLI reading it through `--params path/kernel.toml`? Either choice keeps `render` unchanged; only the source of `ParamSpec` changes.
