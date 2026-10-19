# One render path: Bazel `assemble_asm(defines=)`, registry and tests

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-12 · Size M · Depends on: E05-09 · Behaviour change: none (binaries proven identical by the E02 digests)

## Goal

Every place that turns a templated `.asm` into a program goes through `ipu_as.template.render` via `AssemblySession` (L10):

- Bazel (`assemble_asm(defines=, includes=)`);
- the CLI (E05-10);
- the kernel registry and harness (E06 builds on this);
- tests.

Delete the second sandbox and the unsandboxed `jinja2.Template` uses.

## Background

- `asm_rules.bzl:3-21` (`assemble_asm`) has no `defines` or `includes`. It always runs `ipu-as assemble --input $< --output $@ --format <fmt>`.
- `kernels/convolutions/app.py:128-137` defines `render_asm(text, **context)`, a second `SandboxedEnvironment`. `conv_universal_wide384/app.py:266-267, 330-338` renders per run, assembles into a temp dir, and swaps `self.inst_path`. The three `depthwise_conv_stride2_*` harnesses do the same.
- Six mobilevit tests use unsandboxed `jinja2.Template(...).render(...)`:
  - `test/mobilevit/test_full_layer_l{4,5}_packed.py`;
  - `test_packed_layernorm_{192x64,240x16}.py`;
  - `test_packed_residual_add_{192x64,240x16}.py`.
- Two kernel sources use `| default(...)`: `conv_universal_wide384.asm` and `depthwise_conv_stride2_narrow.asm`.
- `kernel_registry/cases.py:65` assembles through `assemble_to_bin_file(source.read_text(), …)`.

## Implementation

1. **Bazel.** Extend the macro, keeping the CLI flags sorted so the action key is stable:

   ```python
   def assemble_asm(name, src, output_format = "bin", defines = {}, includes = [], **kwargs):
       # defines  -> --define k=v for k in sorted(defines)
       # includes -> added to srcs; -I $$(dirname …) for each distinct directory
   ```

   `ipu_app.bzl:38` passes `includes = native.glob([family + "/**/*.asm.j2"])` and adds the same files to the kernel's `data`.
2. **Declare params** in the two `| default` sources (an `ipu:params` header with today's defaults) and drop the filters. The standalone Bazel target still assembles the default shape, as today.
3. **Registry.** Add `assemble_kernel(source: Path, *, defines: Mapping = {}, search_path=()) -> AssemblyResult` in `kernel_registry/cases.py`, using the session and its cache. `cases.py:65` calls it.
4. **Render-per-run harnesses.** Replace `render_asm` and the `inst_path` swap with `assemble_kernel(..., defines={"cpr": self.cpr, "rows": self.rows})`, and run the in-memory words. Delete `render_asm` from `convolutions/app.py`. E06 later moves this into the runner (`defines(**params)` on the spec); this issue only changes the call.
5. **Mobilevit tests.** Replace `jinja2.Template(...).render(chunk_widths=…)` with `AssemblySession(text, defines={"chunk_widths": …})`. Declare `chunk_widths: tuple[int, ...]` in those templates. The four `asm_packed_output_linear_*` templates then also assemble under the official CLI with `--define chunk_widths=…`.
6. **Lint test.** No `jinja2.Template(`, no `SandboxedEnvironment(` and no `from jinja2` outside `ipu_as/template.py` anywhere under `src/`.

## Tests

- [ ] E02 digests unchanged for every kernel. For the render-per-run kernels, the digests of every `defines` set their cases use are unchanged (E02 records these; add them in this PR if E02 did not).
- [ ] An `assemble_asm(defines={"cpr": "3"})` target builds, and its output equals the CLI with `--define cpr=3`.
- [ ] The mobilevit tests touched here pass. The `test_full_*` manual tests are run by name before and after, and the PR says so.
- [ ] The lint test fails on a planted `jinja2.Template(`.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] One render path. `grep -rn "jinja2" src --include=*.py` matches only `ipu_as/template.py` (and `diagnostics.py` exception types, if still needed).
- [ ] No harness swaps `inst_path` to run a rendered program.

## Risks

- `bazel` `data` omissions: a `*.asm.j2` missing from `data` fails only at run time. `ipu_app.bzl` globs it once, and a test asserts every `{% include %}` target is present in runfiles.
