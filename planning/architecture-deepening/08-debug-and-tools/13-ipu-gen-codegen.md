# Create `ipu_gen`; move the SystemVerilog generator

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-13 · Size M · Depends on: E03 (package split, `ipu_isa` typed view), E02 (SV package golden) · Behaviour change: none (SV package byte-identical)

## Goal

Create the `ipu_gen` package (Q69) at `src/tools/ipu-gen/src/ipu_gen/` and move the SystemVerilog generator into it, with its template and tests. `ipu_gen` reads only `ipu_isa` (plus `ipu_as` where it needs the parser). Nothing else depends on `ipu_gen`. The generator stops deriving output names from assembler class names: it reads them from explicit ISA data.

## Background

- `ipu_as/gen_codegen.py` (459 lines) and `ipu_as/templates/ipu_instr_pkg.sv.j2` produce the vendored RTL package.
  - `gen_codegen.py:17-25` imports `ipu_as.compound_inst`, `ipu_token`, `utils` and `inst.OPERAND_TYPE_MAP`.
  - SV enum typedef names come from token class names (`gen_codegen.py:370` and `EnumToken.get_all_enum_descriptors`).
- Entry points:
  - `docs/gen_codegen_wrapper.py` → `//docs:gen_codegen` → `//docs:generate_instruction_format_artifacts` (`docs/BUILD.bazel:35-51`);
  - `ipu-as sv-package` (`ipu_as/cli.py:53-58`);
  - tests `//src/tools/ipu-as-py:test_gen_codegen` and `:test_sv_package_compiles` (`ipu-as-py/BUILD.bazel:80-100`).
- CLAUDE.md and `docs/content/specs/stage-aaq-str.md:94` reference these targets and the `ipu-as sv-package` command.
- CLAUDE.md requires every `*_WIDTH` `localparam` to stay identical unless the PR calls out a breaking change.

## Implementation

1. Create `src/tools/ipu-gen/` (`BUILD.bazel`, `pyproject.toml`) and `//src/tools/ipu-gen:ipu_gen_lib`, depending on `ipu_isa_lib`. Add `ipu_as_lib` only where a generator needs the parser.
2. `git mv` `gen_codegen.py` → `ipu_gen/sv_package.py` and `templates/ipu_instr_pkg.sv.j2` → `ipu_gen/templates/`. Move the two tests and keep their target names under `//src/tools/ipu-gen:`.
3. Replace class-name-derived SV names with an explicit typedef-name table in the ISA view, mirroring E05-03 for field names. Use an `sv_typedef` attribute per operand type if E03's schema has one, otherwise a table in `ipu_gen` keyed by ISA operand-type names. Assert at import that every operand type has an entry.
4. Add an `ipu-gen` CLI (`//src/tools/ipu-gen:ipu-gen`, a `click` group) with `sv-package --output PATH [--status-file F]`.
   - Remove `sv-package` from `ipu-as`, because `ipu_as` must not import `ipu_gen` (Q62).
   - Point `//docs:generate_instruction_format_artifacts` at `//src/tools/ipu-gen:ipu-gen`, so the target name and output path stay the same.
5. Update the references: CLAUDE.md "Generated SystemVerilog Package" (new test label), `docs/content/specs/stage-aaq-str.md:94`, and `gen_docs.py:531` (the command text).
6. Import-graph test: `ipu_gen` may import `ipu_isa` and `ipu_as`; nothing imports `ipu_gen`.

## Tests

- [ ] E02 SV package golden is byte-identical. Diff every `localparam` line against the base build, as CLAUDE.md requires.
- [ ] `test_sv_package_compiles` (slang, zero diagnostics) and `test_gen_codegen` pass under their new labels.
- [ ] Renaming a token class (monkeypatch `__name__`) does not change the SV output.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] No SV generation code remains in `ipu_as`. `ipu-as --help` has no `sv-package`.
- [ ] `bazel build //docs:generate_instruction_format_artifacts` still writes `bazel-bin/docs/ipu_instr_pkg.sv`.
- [ ] The PR description lists the changed test label for the downstream RTL repo and for CLAUDE.md.
