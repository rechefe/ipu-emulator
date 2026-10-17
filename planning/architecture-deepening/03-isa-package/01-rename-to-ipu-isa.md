# Rename `ipu_common` to `ipu_isa` and move non-ISA modules out

Part of [Epic E03: ISA package](00-epic.md) · ID E03-01 · Size M · Depends on: E02-01, E02-02, E02-03, E02-04 · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

Make the ISA package a package of ISA definitions only (Q13, Q31) with the name and directory Q69 fixes: `src/tools/ipu-isa/src/ipu_isa/`. Move the three non-ISA pieces to the packages that own them, and add the shared error root that E04 and E05 build on. This is a mechanical move; no logic changes.

## Background

`src/tools/ipu-common/src/ipu_common/` (14 modules, 3466 lines) holds the ISA plus three things that are not ISA:

- `activations.py` (137 lines): activation math (`apply_activation`, α and window defaults) used by the emulator (`ipu_emu/ipu.py:56`, `ipu_state.py:17`, `alias_observer.py:8`, `alias_detectors.py:6`). Only `ACTIVATION_FN_NAMES` and the `ACTIVATION_*` ids are ISA: they are the `ActivationFn` operand's enum (`union_layout.py:31,91`, `ipu_as/immediate.py:211-219`).
- `isa_alias_spec.py` (343 lines): MAC-alias catalogue for emulator profiling, used by `ipu_emu/ipu.py:48`, `alias_detectors.py:4` and `ipu-apps/test/test_alias_profile_kernels.py:12`.
- `union_layout_svg.py` + `gen_union_layout_svg.py`: an SVG generator (`ipu-common/BUILD.bazel:19-25`).

`ipu_common` or `ipu-common` appears in 48 files (`.py`, `BUILD.bazel`, `pyproject.toml`, `MODULE.bazel:29,43,53`, docs), including 80 lines of Python.

E04-03 and E05-04 assume an `IpuError` root in `ipu_isa.errors`, "the one package every other package depends on" (Q62).

## Implementation

1. `git mv src/tools/ipu-common src/tools/ipu-isa` and `src/ipu_common` to `src/ipu_isa`. Bazel target `//src/tools/ipu-isa:ipu_isa_lib`. Update `MODULE.bazel`, every `pyproject.toml`, every `BUILD.bazel` dep and every import with one scripted rewrite (commit the script output separately from hand edits). No compatibility alias module (D5: no shims outside the reference API).
2. Split `activations.py`:
   - `ipu_isa/activations.py` keeps `ACTIVATION_*` ids, `ACTIVATION_COUNT` and `ACTIVATION_FN_NAMES` (the operand enum).
   - `ipu_emu/activations.py` gets the math and the α/window defaults.
3. Move `isa_alias_spec.py` to `ipu_emu/isa_alias_spec.py` with its test.
4. Move `union_layout_svg.py`, `gen_union_layout_svg.py` and `test_union_layout_svg.py` to a new `src/tools/ipu-gen/src/ipu_gen/` package with its own `BUILD.bazel` (`ipu_gen_lib`, the `gen_union_layout_svg` binary, the test). E08 later moves the docs, SV and VS Code generators into the same package (cross-epic assumption below).
5. Add `ipu_isa/errors.py`:
   ```python
   class IpuError(Exception):
       """Root of every error raised by IPU tooling packages."""
   ```
   No subclasses here; each package defines its own (E04-03, E05-04).
6. Update `docs/content/adding-instruction.md`, SKILL.md and CLAUDE.md paths (`src/tools/ipu-common/...` to `src/tools/ipu-isa/...`).

## Tests

- [ ] All E02 baselines unchanged: asm digests, decode snapshots, SV golden, generated artifacts.
- [ ] `bazel test //...` green; `bazel test --define=ipu_proto=1 //src/tools/ipu-emu-py/...` green.
- [ ] `grep -rn "ipu_common\|ipu-common" --exclude-dir=bazel-*` finds nothing outside `planning/`.

## Acceptance Criteria

- [ ] `src/tools/ipu-isa/` exists; `src/tools/ipu-common/` does not.
- [ ] `ipu_isa` contains no activation math, no alias catalogue and no SVG code.
- [ ] `ipu_isa.errors.IpuError` exists.

## Risks

- Large but mechanical diff. Land it when few PRs are open, announce it, and keep hand edits in a separate commit for review.
- Cross-epic assumption: E08 adopts `src/tools/ipu-gen/` as the `ipu_gen` package this issue creates, rather than creating a second one.
