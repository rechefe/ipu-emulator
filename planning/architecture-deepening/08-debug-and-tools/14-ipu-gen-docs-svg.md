# Move the docs-page and union-layout SVG generators into `ipu_gen`

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-14 · Size M · Depends on: E08-13 · Behaviour change: none (generated pages and SVGs byte-identical)

## Goal

All generated documentation comes from `ipu_gen`:

- the instruction, operand-type, assembly-syntax, programmer-guide and instruction-format pages;
- the union-layout and struct-layout SVGs.

`ipu_isa` holds only ISA data (Q31: the SVG generators move out). `ipu_as` holds only the assembler.

## Background

- `ipu_as/gen_docs.py` (605 lines) writes the four pages for `//docs:generate_mkdocs_md` (`docs/BUILD.bazel:15-33`) and `instruction-format.md` via `docs/gen_instruction_format_md_wrapper.py` (`docs/BUILD.bazel:53-66`).
- `gen_docs.py:479-497` embeds `CompoundInst.generate_struct_layout_svg()` / `generate_union_layout_svg()` (`compound_inst.py:179-260`). These call `ipu_common.union_layout_svg.render_union_layout_svg`.
- `ipu_common/union_layout_svg.py` and `gen_union_layout_svg.py` (target `//src/tools/ipu-common:gen_union_layout_svg`, test `:test_union_layout_svg`, `ipu-common/BUILD.bazel:20-55`) are generators living in the ISA package.
- `gen_vscode.py:45` imports `OPERAND_TYPE_DETAILS` from `gen_docs`. Under E03 that prose moves into the ISA operand-type `doc` field, so this import changes.

## Implementation

1. `git mv` `ipu_as/gen_docs.py` → `ipu_gen/docs_pages.py`, and `ipu_common/union_layout_svg.py` + `gen_union_layout_svg.py` → `ipu_gen/layout_svg.py`.
2. Move `CompoundInst.generate_*_svg` into `ipu_gen/layout_svg.py`. They read the word layout from the `ipu_isa` codec (`WORD_LAYOUT`, E05-03), not from `CompoundInst`.
3. Add `ipu-gen docs-pages OUT_DIR` and `ipu-gen instruction-format-md OUT`. Point `//docs:generate_mkdocs_md` and `//docs:generate_instruction_format_md` at `//src/tools/ipu-gen:ipu-gen`, keeping the target names and outputs.
4. Move `test_union_layout_svg` and any `gen_docs` tests to `//src/tools/ipu-gen:`.
5. Read `OPERAND_TYPE_DETAILS` from the ISA view (E03's operand-type `doc`). If E03 has not moved it yet, keep it in `docs_pages.py` and have `gen_vscode` import it from there.

## Tests

- [ ] E02-04 generated-docs snapshot is byte-identical: the four pages, `instruction-format.md` and the SVGs.
- [ ] `test_union_layout_svg` passes under its new label.
- [ ] `bazel build //docs:build_docs` succeeds (tagged `docs`, so run it explicitly in the PR).
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] No generator module remains in `ipu_as` or `ipu_isa`. `grep -rln "def generate_\|render_union_layout_svg" src/tools/ipu-as* src/tools/ipu-isa` is empty.
