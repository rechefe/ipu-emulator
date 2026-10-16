# Assemble with case parameters as template values (E05 session)

Part of [Epic: Kernel framework](00-epic.md) · ID E06-07 · Size S · Depends on: E06-03, E05 (strict template render API, assembly session) · Behaviour change: none

## Goal

Assemble a kernel's program for one `CasePlan` through the E05 render and
session API, with the case parameters, the `cr` register names and the format
as template values. This gives one render path, a cached binary per distinct
set of template values, and located errors.

## Background

- There are three render paths today:
  - the assembler (`ipu_as/template.py:42`, no context, no loader);
  - `kernels/convolutions/app.py:128` `render_asm(text, **context)`, a second
    sandbox. Its users swap `self.inst_path` at run time
    (`convolutions/conv_universal_wide384/app.py:329-338`);
  - unsandboxed `jinja2.Template(...)` in 4 mobilevit tests
    (`test/mobilevit/test_full_layer_l4_packed.py:190,231,253,281` and others).
- `| default(...)` hides missing parameters:
  `convolutions/conv_universal_wide384/conv_universal_wide384.asm:128-129`,
  `convolutions/depthwise_conv_stride2_narrow/depthwise_conv_stride2_narrow.asm:3`.
- `assemble_kernel` (`kernel_registry/cases.py:58-66`) assembles the raw
  `.asm`, with no parameters.
- Q80: "assemble with case params as template values (`{{ cr.x }}`)".

## Implementation

Module `ipu_harness/assembly.py`:

```python
def template_values(plan: CasePlan) -> Mapping[str, object]:
    """{**plan.params, "cr": {"x": "CR2", ...}, "format": "e4", "shapes": {...}}"""

def assemble(plan: CasePlan, session: AssemblySession) -> Program:
    """Render <name>.asm with template_values(plan) and assemble it (E05).
    The session caches by (source digest, include digests, values)."""
```

- The declared template parameters are the `kernel.toml` `[params]` names plus
  the reserved names `cr`, `lr`, `format` and `shapes` (`cr` and `lr` are the
  `[abi]` register names, passed as the render's namespaces). The framework passes them to
  the E05 render API as the declaration set, so a kernel does not repeat its
  parameters in an `ipu:params` header. `StrictUndefined` turns any other name
  into a located error (`scale.asm:9: undefined 'cr.z'`).
- The include search path is the kernel directory, the family directory, and
  the shared macro library `ipu_kernels/asm_lib/` (`*.asm.j2`).
- `Program` is the E05 assembled artefact (instruction words plus the source
  map). The runner (E06-11) loads it through the E04 API with no temp-file
  round-trip.
- Bazel: `ipu_kernel()` (E06-13) also declares `assemble_<name>`, which renders
  with the default case's values. It keeps the existing `assemble_*` targets
  and labels that the E02 asm digests cover, and feeds `--target hardware` (E05).

## Tests

- [ ] `scale` renders `{{ cr.x }}` as `CR2` and assembles identically in all three formats (its program does not use `format`).
- [ ] An undefined template name and an undeclared parameter each give a located error.
- [ ] Two cases with the same template values assemble once (a session cache hit). Different `rows` values assemble once each, but only if the template uses `rows`.
- [ ] `assemble_<name>` output equals the binary from the E02 asm digest for a migrated kernel (exercised by E07).
- [ ] Assembly takes a `CasePlan` and re-checks nothing (E06-03's single seam).

## Acceptance Criteria

- [ ] `ipu_harness` renders only through the E05 render API. No `jinja2` import exists in `ipu_harness` or `ipu_kernels`.
- [ ] `bazel test //...` is green.
