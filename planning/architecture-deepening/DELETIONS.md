# Deletions in the architecture-deepening PR

Every file the PR deletes (renames excluded), why, and what replaced it. `tools/check_deletion_log.sh` fails while a deleted file is missing here. Kernel-migration (E07) rows also give the equivalence evidence: the `compare_legacy` result and the E02 baseline digest.

## E00 Verified fixes

| Deleted | Issue | Why | Replaced by | Evidence |
|---|---|---|---|---|
| `tools/fp32_to_fp4.py` | E00-10 (87e6aa20) | Unused and wrong. It encoded 4.0 as 6.0 and 0.5 as 0, and raised on `inf`. Decision L41 (accepted via Q29) deletes it. | — (nothing: dead code) | No BUILD file, script or doc named it on the base (`grep -rn fp32_to_fp4` found only the file). `git grep fp32_to_fp4 HEAD` outside `planning/` finds nothing. Out-of-tree callers, if any, break; the issue asks the PR description to say so. |

## E01 Foundation

| Deleted | Issue | Why | Replaced by | Evidence |
|---|---|---|---|---|
| `tools/python_deps.bzl` | E01-02 (417cb4ce) | It held the `generate_requirements` repository rule, which ran uv and resolved against live PyPI at fetch time. The build now installs from a committed, hash-pinned lock. | `requirements_lock.txt` (read by `pip.parse` in `MODULE.bazel`), refreshed by `//tools:update_requirements_lock` (`tools/update_requirements_lock.py`) | The first lock used the old resolved versions as constraints: same 43 packages, plus uv. The refresh tool reproduces the lock byte for byte on an unchanged tree. |
| `src/tools/ipu-as-py/uv.lock` | E01-02 (417cb4ce) | Unused and stale. Nothing read it; the build resolved through `tools/python_deps.bzl`. | `requirements_lock.txt` | Commit message: "Deleted the unused, stale ... uv.lock". |
| `src/tools/ipu-emu-py/uv.lock` | E01-02 (417cb4ce) | Unused and stale, as above. | `requirements_lock.txt` | As above. |
| `tools/pyproject_deps.bzl` | E01-03 (564cc55b) | It parsed each `pyproject.toml` into a repository (`parse_pyproject_deps`), so libraries pulled in test and docs packages. Each package now lists its runtime deps once. | `RUNTIME_DEPS` in each package's `deps.bzl` (for example `src/tools/ipu-as-py/deps.bzl`), `tools/runtime_deps.bzl`, and `//tools:deps_consistency_test` (`tools/deps_consistency_test.py`) | The three `parse_pyproject_deps` repositories left `MODULE.bazel` in the same commit. The consistency test checks each `RUNTIME_DEPS` against `[project].dependencies` and the lock. |

## E03 ISA package

E03-01 renamed `src/tools/ipu-common/` to `src/tools/ipu-isa/` with `git mv` (70e25009), then moved the non-ISA modules out (173010e0). Git sees the files below as deletions, not renames, because later E03 issues changed them too much for rename detection. No compatibility alias is kept (decision D5).

| Deleted | Issue | Why | Replaced by | Evidence |
|---|---|---|---|---|
| `src/tools/ipu-common/BUILD.bazel` | E03-01 (70e25009) | Package renamed to `ipu_isa` (Q69). | `src/tools/ipu-isa/BUILD.bazel` (`//src/tools/ipu-isa:ipu_isa_lib`) | Scripted `git mv` in 70e25009; later issues added targets (235 lines added since). |
| `src/tools/ipu-common/README.md` | E03-01 (70e25009) | Package renamed. | `src/tools/ipu-isa/README.md` | Scripted `git mv` in 70e25009. |
| `src/tools/ipu-common/src/ipu_common/__init__.py` | E03-01 (70e25009) | Package renamed. | `src/tools/ipu-isa/src/ipu_isa/__init__.py` | Scripted `git mv`; every import rewritten by `sed`. |
| `src/tools/ipu-common/src/ipu_common/lr_inc_dec_imm.py` | E03-01 (70e25009) | Package renamed. | `src/tools/ipu-isa/src/ipu_isa/lr_inc_dec_imm.py` | Scripted `git mv`. |
| `src/tools/ipu-common/src/ipu_common/union_layout.py` | E03-01 (70e25009) | Package renamed. | `src/tools/ipu-isa/src/ipu_isa/union_layout.py` | Scripted `git mv`; later E03 issues rewrote much of it (166 lines added, 134 removed since). |
| `src/tools/ipu-common/src/ipu_common/union_layout_svg.py` | E03-01 (70e25009) | An SVG generator is not ISA data. 173010e0 moved it to the new `ipu_gen` package. E08-14 (393fe045) then renamed it and made it read the ISA instead of `CompoundInst`. | `src/tools/ipu-gen/src/ipu_gen/layout_svg.py` | Generated SVGs are byte-identical (E08-14 commit message). |
| `src/tools/ipu-common/src/ipu_common/gen_union_layout_svg.py` | E03-01 (70e25009) | Moved to `ipu_gen` (173010e0). E08-14 (393fe045) folded it into the `ipu-gen` CLI and deleted it. | `ipu-gen union-layout-svg` in `src/tools/ipu-gen/src/ipu_gen/cli.py` (replaces the `//src/tools/ipu-gen:gen_union_layout_svg` target) | Behaviour change named in the E08-14 commit: the target became a subcommand. |

## E05 Assembler

| Deleted | Issue | Why | Replaced by | Evidence |
|---|---|---|---|---|
| `src/tools/ipu-as-py/src/ipu_as/label.py` | E05-05 (1ee96a69) | It held one global label table and `reset_labels()`, so two assemblies in one process shared labels. Each assembly now has its own table. | `src/tools/ipu-as-py/src/ipu_as/symbols.py` (`SymbolTable`, same error messages), owned by `AssemblySession` in `src/tools/ipu-as-py/src/ipu_as/session.py` | `git grep reset_labels HEAD -- src` finds nothing. `//src/tools/ipu-as-py:test_session` tests interleaved sessions. |

## E08 Debug and tools

| Deleted | Issue | Why | Replaced by | Evidence |
|---|---|---|---|---|
| `src/tools/ipu-emu-py/src/ipu_emu/debug_cli.py` | E08-03 (5b06f6fd) | The debugger left `ipu_emu` for its own package, so the emulator does not depend on it. | `src/tools/ipu-debug/src/ipu_debug/cli.py`; E08-04 moved its parse, resolve and read API to `src/tools/ipu-debug/src/ipu_debug/session.py` (`DebugSession`) | `git mv` in 5b06f6fd. `//tools:test_import_graph` checks that only `ipu_apps` imports `ipu_debug`. |
| `src/tools/ipu-emu-py/src/ipu_emu/debug_control.py` | E08-03 (5b06f6fd) | As above. `RunControl` stays in `ipu_emu`. | `src/tools/ipu-debug/src/ipu_debug/control.py` | `git mv` in 5b06f6fd. |
| `src/tools/ipu-emu-py/src/ipu_emu/debug_launch.py` | E08-03 (5b06f6fd) | As above. | `src/tools/ipu-debug/src/ipu_debug/launch.py` | `git mv` in 5b06f6fd; `ipu_apps` imports `ipu_debug.launch`. |
| `src/tools/ipu-emu-py/src/ipu_emu/debug_pipeline.py` | E08-03 (5b06f6fd) | As above. | `src/tools/ipu-debug/src/ipu_debug/pipeline.py` | `git mv` in 5b06f6fd. |
| `src/tools/ipu-emu-py/src/ipu_emu/debug_tui.py` | E08-03 (5b06f6fd) | As above. E08-08 to E08-11 then split the one module into a package. | `src/tools/ipu-debug/src/ipu_debug/tui/` (`view.py`, `screen.py`, `keymap.py`, `panes/`, and others) | `git mv` in 5b06f6fd; split in 6896606f (E08-08) and c2a846dd (E08-11). E08-11: golden screens in `test_tui_shell` unchanged. |
| `src/tools/ipu-emu-py/test/test_debug_tui.py` | E08-03 (5b06f6fd) | Moved with the debugger. Later E08 issues moved some tests to per-pane test files. | `src/tools/ipu-debug/test/test_debug_tui.py` (`//src/tools/ipu-debug:test_debug_tui`), plus `test_tui_disassembly_pane.py`, `test_tui_pipeline_pane.py`, `test_tui_shell.py` in the same directory | Same 276 test cases right after the move (5b06f6fd). Every test is mapped below. |
| `src/tools/ipu-as-py/test/test_gen_codegen.py` | E08-13 (c59058c4) | The SV package generator moved from `ipu_as` to `ipu_gen` (Q69), and its test moved with it under the same target name. | `src/tools/ipu-gen/test/test_gen_codegen.py` (`//src/tools/ipu-gen:test_gen_codegen`) | `git mv` in c59058c4. The SV package is byte-identical; no localparam changed. |
| `docs/gen_codegen_wrapper.py` | E08-13 (c59058c4) | A shim that called `ipu_as.gen_codegen.generate_sv_package`. The generator now has a CLI. | `ipu-gen sv-package` (`src/tools/ipu-gen/src/ipu_gen/cli.py`, generator in `src/tools/ipu-gen/src/ipu_gen/sv_package.py`); `//docs:generate_instruction_format_artifacts` runs it | Same target, same output (c59058c4). |
| `src/tools/ipu-as-py/src/ipu_as/gen_docs.py` | E08-14 (393fe045) | All generated docs come from `ipu_gen`, so `ipu_as` no longer imports `ipu_gen`. | `src/tools/ipu-gen/src/ipu_gen/docs_pages.py` | `git mv` in 393fe045. Only `operand-types.md` changed, by the one line that names the generator module. |
| `docs/gen_docs_wrapper.py` | E08-14 (393fe045) | A shim that called `ipu_as.gen_docs.generate_all_docs`. | `ipu-gen docs-pages` (`src/tools/ipu-gen/src/ipu_gen/cli.py`); `//docs:generate_mkdocs_md` runs it | Same target, same outputs (393fe045). |
| `docs/gen_instruction_format_md_wrapper.py` | E08-14 (393fe045) | A shim that called `ipu_as.gen_docs.generate_instruction_format_md`. | `ipu-gen instruction-format-md` (`src/tools/ipu-gen/src/ipu_gen/cli.py`); `//docs:generate_instruction_format_md` runs it | Same target, same output; `instruction-format.md` byte-identical (393fe045). |
| `src/tools/ipu-as-py/test/test_isa_data.py` | E08-15 (6b706e98) | It tests the VS Code data generator, which moved to `ipu_gen`. | `src/tools/ipu-gen/test/test_isa_data.py` (`//src/tools/ipu-gen:test_isa_data`) | `git mv` in 6b706e98. |
| `docs/mkdocs_wrapper.py` | E08-15 (6b706e98) | A shim around `mkdocs.__main__.cli`. No `*_wrapper.py` shims are left. | `//docs:mkdocs`, a `py_console_script_binary` in `docs/BUILD.bazel` | `--config=docs //docs:build_docs` builds with it (6b706e98). |
| `vscode-ipu-asm/gen_vscode_wrapper.py` | E08-15 (6b706e98) | A shim that called `ipu_as.gen_vscode` and read the emulator's private `Ipu._lrd_lr_indices`. | `//vscode-ipu-asm:gen_vscode`, an alias to `//src/tools/ipu-gen:vscode_data` (`ipu-gen vscode-data`, generator `src/tools/ipu-gen/src/ipu_gen/vscode_data.py`, register facts in its `REGISTER_FACTS`, LRD pairs from `ipu_isa.registers.lrd_pair`) | Grammar, `language-configuration.json` and `isa.json` byte-identical (E02-04 snapshot passes). |
| `vscode-ipu-asm/dump_parser_tokens_wrapper.py` | E08-15 (6b706e98) | A shim that called `ipu_as.dump_parser_tokens.main`. | `//vscode-ipu-asm:dump_parser_tokens`, an alias to the `py_binary` `//src/tools/ipu-as-py:dump_parser_tokens` (`src/tools/ipu-as-py/src/ipu_as/dump_parser_tokens.py`) | `test/language.js` passes through the alias (14 cases, 6b706e98). |

## E09 Docs

| Deleted | Issue | Why | Replaced by | Evidence |
|---|---|---|---|---|
| `TODO.md` | E09-03 (b93dfd65) | Its items are done or tracked elsewhere. | — (each item placed, see Evidence) | Softmax kernel, CNN kernel, LR union: done (`src/tools/ipu-apps/src/ipu_apps/kernels/softmax/`, `src/tools/ipu-apps/src/ipu_apps/kernels/convolutions/`, `src/tools/ipu-isa/src/ipu_isa/union_layout.py`). "Implement quantization": open question OQ-20 in `docs/content/requirements/open-questions.md`. "Add preprocessing CLI options": E05-10 (`--define`). "Add Metadata to registers and XMEM": not tracked anywhere; the commit leaves it to the owner to decide. |
| `docs/content/setup.md` | E09-08 (58271b18) | Replaced by the new Getting started page. Its GCC, C/C++, CLion and local uv content no longer applies. | `docs/content/getting-started/index.md`; lint and type-check sections were already in `docs/content/code-conventions.md` | Redirected to Getting started in `docs/mkdocs.yml`; the nav entry was removed. |
| `docs/content/adding-instruction.md` | E09-11 (01e17e08) | Replaced by a shorter how-to that shows the three edits the "one edit" test proves are enough. | `docs/content/how-to/add-an-instruction.md`, with examples tested by `//docs:test_add_an_instruction` | Inbound links fixed in the same commit. Redirected in `docs/mkdocs.yml`. |
| `docs/content/debugging.md` | E09-12 (b0e1c9a1) | 681 lines, mostly about the deprecated line debugger. Replaced by a how-to organised by symptom. | `docs/content/how-to/debug-a-kernel.md`; key bindings in the generated table in `docs/content/reference/cli.md` | Redirect in `docs/mkdocs.yml`. The deprecated line debugger is no longer documented. |
| `docs/content/ipu-configuration.md` | E09-14 (59f94af3) | Hand-written reference replaced by a page generated from `ipu_isa.constants`. The old page had wrong field bits (partition [11:8], pad_mode [13:12]; correct: [12:8], [14:13]). | Generated `reference/isa/configuration-registers.md` (from `src/tools/ipu-gen/src/ipu_gen/docs_pages.py`); element-format text in `docs/content/explanation/element-formats.md` | Redirect in `docs/mkdocs.yml`. Baseline `configuration-registers.md` added under `src/tools/ipu-as-py/test/baselines/generated/`. |
| `docs/content/prototyping-mode.md` | E09-17 (da88ee12) | Folded into the architecture explanation page. | `docs/content/explanation/architecture.md` | Redirect in `docs/mkdocs.yml`. The closed E04-07 divergence was dropped. |
| `docs/content/wide-vector-debug-mode.md` | E09-19 (858c86b7) | Folded into the element-formats explanation page. | `docs/content/explanation/element-formats.md` | Redirect in `docs/mkdocs.yml`. Its 4-byte alignment rule was stale (E04-15 made `rc_idx` an element index) and was dropped. |

### Tests from `src/tools/ipu-emu-py/test/test_debug_tui.py`

All 189 test functions. 166 keep their name. The 23 others were renamed when E08-08 (6896606f), E08-11 (c2a846dd) and E08-12 (b4f08709) rewrote the TUI. E08-11 says "test_debug_tui loses 23 tests to test_tui_disassembly_pane and test_tui_pipeline_pane". Renamed rows were matched by reading both test bodies. None was dropped.

| Test | Now covered by |
|---|---|
| `test_minimum_terminal_has_all_panes` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_minimum_terminal_has_all_panes` |
| `test_small_terminal_shows_resize_message` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_small_terminal_shows_resize_message` |
| `test_render_shows_status_registers_disassembly_and_xmem` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_render_shows_status_registers_disassembly_and_xmem` |
| `test_register_columns_are_complete_and_separate` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_register_columns_are_complete_and_separate` |
| `test_register_columns_never_touch_each_other` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_register_columns_never_touch_each_other` |
| `test_wide_terminal_labels_scalar_register_groups` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_wide_terminal_labels_scalar_register_groups` |
| `test_narrow_register_pane_scrolls_columns_instead_of_crowding` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_narrow_register_pane_scrolls_columns_instead_of_crowding` |
| `test_parallel_operations_are_stacked_under_their_instruction` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestListing::test_parallel_operations_are_stacked_under_their_instruction` |
| `test_no_instruction_spans_the_pane_sideways` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestListing::test_no_instruction_spans_the_pane_sideways` |
| `test_every_row_of_an_instruction_carries_its_cursor` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestCursorAndReadout::test_every_row_of_the_cursor_instruction_is_highlighted` (renamed) |
| `test_mnemonics_share_one_column_across_every_slot` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestListing::test_mnemonics_share_one_column_across_slots_and_instructions` (renamed) |
| `test_the_parallel_marker_recedes` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestListing::test_the_parallel_marker_recedes` |
| `test_operands_line_up_under_a_padded_mnemonic` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestListing::test_mnemonics_share_one_column_across_slots_and_instructions` (renamed) |
| `test_nops_are_collapsed_in_disassembly` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestListing::test_compact_leaves_nops_out_and_full_shows_every_slot` (renamed) |
| `test_tab_bar_scrolls_to_keep_active_tab_visible` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_tab_bar_scrolls_to_keep_active_tab_visible` |
| `test_tab_bar_marks_tabs_scrolled_out_of_view` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_tab_bar_marks_tabs_scrolled_out_of_view` |
| `test_only_focused_pane_uses_selected_tab_color` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_only_focused_pane_uses_selected_tab_color` |
| `test_minimum_footer_uses_readable_contextual_labels` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_minimum_footer_uses_readable_contextual_labels` |
| `test_wide_footer_includes_global_controls` | `src/tools/ipu-debug/test/test_debug_tui.py::TestLayout::test_wide_footer_includes_global_controls` |
| `test_shared_edges_resolve_into_junctions` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPaneFrames::test_shared_edges_resolve_into_junctions` |
| `test_focused_pane_owns_the_attribute_of_a_shared_edge` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPaneFrames::test_focused_pane_owns_the_attribute_of_a_shared_edge` |
| `test_ascii_glyphs_stay_ascii` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPaneFrames::test_ascii_glyphs_stay_ascii` |
| `test_glyph_set_follows_the_terminal_encoding` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPaneFrames::test_glyph_set_follows_the_terminal_encoding` |
| `test_whole_segments_are_dropped_rather_than_cut` | `src/tools/ipu-debug/test/test_debug_tui.py::TestTitleFitting::test_whole_segments_are_dropped_rather_than_cut` |
| `test_squeezed_pane_keeps_its_name_and_drops_the_readout` | `src/tools/ipu-debug/test/test_debug_tui.py::TestTitleFitting::test_squeezed_pane_keeps_its_name_and_drops_the_readout` |
| `test_pane_shows_both_the_title_and_the_readout_when_they_fit` | `src/tools/ipu-debug/test/test_debug_tui.py::TestTitleFitting::test_pane_shows_both_the_title_and_the_readout_when_they_fit` |
| `test_default_tab_is_one_full_active_row` | `src/tools/ipu-debug/test/test_debug_tui.py::TestSessionPersistence::test_default_tab_is_one_full_active_row` |
| `test_the_session_passed_to_each_stop_persists` | `src/tools/ipu-debug/test/test_debug_tui.py::TestSessionPersistence::test_the_session_passed_to_each_stop_persists` |
| `test_step_keeps_view_active_and_captures_baseline` | `src/tools/ipu-debug/test/test_debug_tui.py::TestSessionPersistence::test_step_keeps_view_active_and_captures_baseline` |
| `test_comparison_baselines_are_stored_per_tab` | `src/tools/ipu-debug/test/test_debug_tui.py::TestSessionPersistence::test_comparison_baselines_are_stored_per_tab` |
| `test_new_tab_has_no_stale_comparison_baseline` | `src/tools/ipu-debug/test/test_debug_tui.py::TestSessionPersistence::test_new_tab_has_no_stale_comparison_baseline` |
| `test_escape_does_not_leave_tui` | `src/tools/ipu-debug/test/test_debug_tui.py::TestSessionPersistence::test_escape_does_not_leave_tui` |
| `test_printable_input_is_ignored_outside_xmem_editor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDedicatedControls::test_printable_input_is_ignored_outside_xmem_editor` |
| `test_f5_f8_and_q_are_direct_execution_controls` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDedicatedControls::test_f5_f8_and_q_are_direct_execution_controls` |
| `test_execution_keys_work_without_changing_focus` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDedicatedControls::test_execution_keys_work_without_changing_focus` |
| `test_xmem_editor_supports_cursor_editing_and_cancel` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDedicatedControls::test_xmem_editor_supports_cursor_editing_and_cancel` |
| `test_f2_adds_valid_symbolic_xmem_tab` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_f2_adds_valid_symbolic_xmem_tab` |
| `test_invalid_tab_request_stays_in_editor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_invalid_tab_request_stays_in_editor` |
| `test_edit_replaces_active_tab_and_f4_closes_it` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_edit_replaces_active_tab_and_f4_closes_it` |
| `test_number_keys_focus_a_pane_directly` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_number_keys_focus_a_pane_directly` |
| `test_number_keys_inside_an_editor_are_typed_text` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_number_keys_inside_an_editor_are_typed_text` |
| `test_shift_tab_cycles_pane_focus_and_tab_does_not` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_shift_tab_cycles_pane_focus_and_tab_does_not` |
| `test_tab_cycles_xmem_tabs` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_tab_cycles_xmem_tabs` |
| `test_arrows_move_xmem_cursor_within_active_tab` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_arrows_move_xmem_cursor_within_active_tab` |
| `test_xmem_cursor_is_highlighted` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_xmem_cursor_is_highlighted` |
| `test_cell16_selection_preserves_encoded_color_pair` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_cell16_selection_preserves_encoded_color_pair` |
| `test_cell16_title_reports_colors_encoded_in_memory` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_cell16_title_reports_colors_encoded_in_memory` |
| `test_color_manager_allocates_all_16_cell_colors` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_palette_allocates_all_16_cell_colors` (renamed) |
| `test_changed_register_and_xmem_value_are_highlighted` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_changed_register_and_xmem_value_are_highlighted` |
| `test_all_xmem_formats_render` | `src/tools/ipu-debug/test/test_debug_tui.py::TestXmemTabs::test_all_xmem_formats_render` |
| `test_viewer_covers_every_logical_vector_register` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_the_registers_are_every_logical_vector_register` (renamed) |
| `test_every_logical_pipeline_register_is_selectable` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_every_logical_pipeline_register_is_selectable` |
| `test_a_d_select_tabs_without_changing_focus` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_a_d_select_tabs_without_changing_focus` |
| `test_f2_and_f3_open_editor_for_focused_pane` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_f2_and_f3_open_editor_for_focused_pane` |
| `test_add_edit_and_close_tabs_for_each_non_xmem_pane` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_add_edit_and_close_tabs_for_each_non_xmem_pane` |
| `test_invalid_pane_editor_input_does_not_change_tabs` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_invalid_pane_editor_input_does_not_change_tabs` |
| `test_disassembly_editor_accepts_current_and_valid_fixed_pc` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestEditors::test_the_tab_editor_accepts_current_and_a_fixed_pc` (renamed) |
| `test_disassembly_editor_rejects_out_of_range_pc` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestEditors::test_the_tab_editor_rejects_what_is_not_a_pc` (renamed) |
| `test_register_tab_filters_rendered_group` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_register_tab_filters_rendered_group` |
| `test_tab_selects_next_tab_in_every_pane` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_tab_selects_next_tab_in_every_pane` |
| `test_scalar_register_format_cycles_and_renders` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_scalar_register_format_cycles_and_renders` |
| `test_register_format_is_per_tab` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_register_format_is_per_tab` |
| `test_pipeline_register_format_cycles_and_renders` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_cycling_the_format_walks_every_format_and_resets_the_cursor` (renamed) |
| `test_pipeline_cell16_preserves_color_and_groups_characters` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_cell16_keeps_each_cells_colours_and_groups_characters` (renamed) |
| `test_unsigned_register_formats_use_xmem_zero_padding` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_unsigned_register_formats_use_xmem_zero_padding` |
| `test_leading_zeros_of_a_wide_value_are_dimmed` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_leading_zeros_of_a_wide_value_are_dimmed` |
| `test_wide_pipeline_hex_groups_every_four_bytes` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_wide_hex_groups_every_four_bytes` (renamed) |
| `test_pipeline_format_is_per_tab` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_pipeline_format_is_per_tab` |
| `test_xmem_format_cycle_preserves_byte_range_and_skips_alignment` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_xmem_format_cycle_preserves_byte_range_and_skips_alignment` |
| `test_xmem_format_cycle_preserves_aligned_resolved_byte_range` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_xmem_format_cycle_preserves_aligned_resolved_byte_range` |
| `test_disassembly_format_is_per_tab` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestEditors::test_cycling_the_format_walks_every_format_per_tab` (renamed) |
| `test_wide_r0_uses_wide_storage_and_arithmetic_format` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_wide_mode_reads_the_wide_storage_in_the_arithmetic_format` (renamed) |
| `test_pipeline_changes_are_highlighted` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_values_changed_since_the_baseline_are_marked` (renamed) |
| `test_pipeline_arrows_move_value_cursor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_pipeline_arrows_move_value_cursor` |
| `test_scalar_register_arrows_move_cursor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_scalar_register_arrows_move_cursor` |
| `test_disassembly_focus_owns_keyboard_scrolling` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineRegisters::test_disassembly_focus_owns_keyboard_scrolling` |
| `test_every_pane_shows_its_cursor_whether_focused_or_not` | `src/tools/ipu-debug/test/test_debug_tui.py::TestCursorVisibility::test_every_pane_shows_its_cursor_whether_focused_or_not` |
| `test_the_focused_and_idle_cursors_are_distinguishable` | `src/tools/ipu-debug/test/test_debug_tui.py::TestCursorVisibility::test_the_focused_and_idle_cursors_are_distinguishable` |
| `test_an_instruction_is_decoded_once_per_curses_entry` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestListing::test_an_instruction_is_decoded_once_per_pane` (renamed) |
| `test_an_xmem_request_is_tokenised_once_per_curses_entry` | `src/tools/ipu-debug/test/test_debug_tui.py::TestRenderCost::test_an_xmem_request_is_tokenised_once_per_curses_entry` |
| `test_bare_pointer_motion_does_not_ask_for_a_redraw` | `src/tools/ipu-debug/test/test_debug_tui.py::TestRenderCost::test_bare_pointer_motion_does_not_ask_for_a_redraw` |
| `test_queued_input_is_handled_before_the_next_frame` | `src/tools/ipu-debug/test/test_debug_tui.py::TestRenderCost::test_queued_input_is_handled_before_the_next_frame` |
| `test_clicking_an_xmem_value_moves_the_cursor_to_it` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMousePlacesTheCursor::test_clicking_an_xmem_value_moves_the_cursor_to_it` |
| `test_clicking_a_register_moves_the_cursor_to_it` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMousePlacesTheCursor::test_clicking_a_register_moves_the_cursor_to_it` |
| `test_clicking_an_instruction_moves_the_disassembly_cursor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMousePlacesTheCursor::test_clicking_an_instruction_moves_the_disassembly_cursor` |
| `test_clicking_a_pipeline_value_moves_the_cursor_to_it` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMousePlacesTheCursor::test_clicking_a_pipeline_value_moves_the_cursor_to_it` |
| `test_wheel_scrolls_by_several_lines` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMousePlacesTheCursor::test_wheel_scrolls_by_several_lines` |
| `test_clicking_the_scrollbar_jumps_through_the_content` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMousePlacesTheCursor::test_clicking_the_scrollbar_jumps_through_the_content` |
| `test_dragging_a_vertical_border_moves_that_split` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseResizesSplits::test_dragging_a_vertical_border_moves_that_split` |
| `test_dragging_the_shared_row_moves_the_top_split` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseResizesSplits::test_dragging_the_shared_row_moves_the_top_split` |
| `test_a_press_and_release_without_motion_is_a_click` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseResizesSplits::test_a_press_and_release_without_motion_is_a_click` |
| `test_a_keystroke_abandons_an_unfinished_drag` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseResizesSplits::test_a_keystroke_abandons_an_unfinished_drag` |
| `test_pointer_motion_alone_changes_nothing` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseResizesSplits::test_pointer_motion_alone_changes_nothing` |
| `test_mouse_selects_pipeline_tab` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseAndTerminalFallbacks::test_mouse_selects_pipeline_tab` |
| `test_mouse_click_focuses_pipeline_pane` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseAndTerminalFallbacks::test_mouse_click_focuses_pipeline_pane` |
| `test_mouse_click_on_add_control_opens_focused_editor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseAndTerminalFallbacks::test_mouse_click_on_add_control_opens_focused_editor` |
| `test_mouse_wheel_moves_cursor_in_pointed_pane` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseAndTerminalFallbacks::test_mouse_wheel_moves_cursor_in_pointed_pane` |
| `test_run_debug_tui_uses_wrapper_and_returns_action` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseAndTerminalFallbacks::test_run_debug_tui_uses_the_terminal_and_returns_action` (renamed) |
| `test_curses_initialization_failure_is_reported` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseAndTerminalFallbacks::test_curses_initialization_failure_is_logged_and_raised` (renamed) |
| `test_wrapper_restores_terminal_when_view_raises` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseAndTerminalFallbacks::test_wrapper_restores_terminal_when_view_raises` |
| `test_narrow_footer_keeps_whole_labels_quit_and_help` | `src/tools/ipu-debug/test/test_debug_tui.py::TestFooterFitting::test_narrow_footer_keeps_whole_labels_quit_and_help` |
| `test_footer_labels_come_from_the_binding_table` | `src/tools/ipu-debug/test/test_debug_tui.py::TestFooterFitting::test_footer_labels_come_from_the_binding_table` |
| `test_footer_never_splits_the_tab_selection_labels` | `src/tools/ipu-debug/test/test_debug_tui.py::TestFooterFitting::test_footer_never_splits_the_tab_selection_labels` |
| `test_zero_values_keep_the_selection_color_pair` | `src/tools/ipu-debug/test/test_debug_tui.py::test_zero_values_keep_the_selection_color_pair` |
| `test_home_and_end_jump_to_the_first_and_last_value` | `src/tools/ipu-debug/test/test_debug_tui.py::TestJumpKeys::test_home_and_end_jump_to_the_first_and_last_value` |
| `test_end_selects_the_last_scalar_register` | `src/tools/ipu-debug/test/test_debug_tui.py::TestJumpKeys::test_end_selects_the_last_scalar_register` |
| `test_end_selects_the_last_pipeline_item` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_arrows_move_by_values_and_lines_and_stop_at_the_ends` (renamed) |
| `test_xmem_home_resets_both_cursor_axes` | `src/tools/ipu-debug/test/test_debug_tui.py::TestJumpKeys::test_xmem_home_resets_both_cursor_axes` |
| `test_question_mark_opens_a_scrollable_overlay` | `src/tools/ipu-debug/test/test_debug_tui.py::TestHelpOverlay::test_question_mark_opens_a_scrollable_overlay` |
| `test_overlay_lists_every_documented_binding` | `src/tools/ipu-debug/test/test_debug_tui.py::TestHelpOverlay::test_overlay_lists_every_documented_binding` |
| `test_overlay_lines_fit_the_minimum_terminal` | `src/tools/ipu-debug/test/test_debug_tui.py::TestHelpOverlay::test_overlay_lines_fit_the_minimum_terminal` |
| `test_overlay_swallows_execution_keys` | `src/tools/ipu-debug/test/test_debug_tui.py::TestHelpOverlay::test_overlay_swallows_execution_keys` |
| `test_question_mark_inside_an_editor_is_typed_text` | `src/tools/ipu-debug/test/test_debug_tui.py::TestHelpOverlay::test_question_mark_inside_an_editor_is_typed_text` |
| `test_escape_closes_the_overlay_before_the_editor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestHelpOverlay::test_escape_closes_the_overlay_before_the_editor` |
| `test_pipeline_reports_the_visible_item_range` | `src/tools/ipu-debug/test/test_debug_tui.py::TestScrollPosition::test_pipeline_reports_the_visible_item_range` |
| `test_scroll_thumb_tracks_the_cursor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestScrollPosition::test_scroll_thumb_tracks_the_cursor` |
| `test_thumb_stays_clear_of_the_tab_bar_and_borders` | `src/tools/ipu-debug/test/test_debug_tui.py::TestScrollPosition::test_thumb_stays_clear_of_the_tab_bar_and_borders` |
| `test_no_thumb_when_everything_fits` | `src/tools/ipu-debug/test/test_debug_tui.py::TestScrollPosition::test_no_thumb_when_everything_fits` |
| `test_every_scrollable_pane_draws_an_inset_thumb` | `src/tools/ipu-debug/test/test_debug_tui.py::TestScrollPosition::test_every_scrollable_pane_draws_an_inset_thumb` |
| `test_xmem_reports_its_line_range` | `src/tools/ipu-debug/test/test_debug_tui.py::TestScrollPosition::test_xmem_reports_its_line_range` |
| `test_shift_arrows_resize_and_persist_in_the_session` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_shift_arrows_resize_and_persist_in_the_session` |
| `test_shift_arrows_resize_the_focused_row_only` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_shift_arrows_resize_the_focused_row_only` |
| `test_resizing_preserves_the_shared_border_columns` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_resizing_preserves_the_shared_border_columns` |
| `test_splits_are_clamped_to_their_minimums` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_splits_are_clamped_to_their_minimums` |
| `test_bottom_split_is_clamped_to_its_minimums` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_bottom_split_is_clamped_to_its_minimums` |
| `test_top_pane_shrinks_and_the_register_grid_reflows` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_top_pane_shrinks_and_the_register_grid_reflows` |
| `test_top_height_keeps_instruction_width_independent_of_register_grid` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_top_height_keeps_instruction_width_independent_of_register_grid` |
| `test_the_default_top_pane_fits_the_tallest_register_grid` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_the_default_top_pane_fits_the_tallest_register_grid` |
| `test_growing_the_top_pane_leaves_room_for_memory` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_growing_the_top_pane_leaves_room_for_memory` |
| `test_equals_resets_both_splits` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResizableSplits::test_equals_resets_both_splits` |
| `test_queued_maximize_focus_and_navigation_use_current_geometry` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_queued_maximize_focus_and_navigation_use_current_geometry` |
| `test_click_after_tab_close_uses_updated_targets` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_click_after_tab_close_uses_updated_targets` |
| `test_idle_mouse_preserves_pending_keyboard_redraw` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_idle_mouse_preserves_pending_keyboard_redraw` |
| `test_queued_step_does_not_consume_following_input` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_queued_step_does_not_consume_following_input` |
| `test_mouse_payload_is_read_before_next_queued_key` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_queued_mouse_event_is_handled_before_the_next_key` (renamed) |
| `test_stationary_motion_keeps_drag_until_release` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_stationary_motion_keeps_drag_until_release` |
| `test_small_terminal_discards_old_mouse_targets` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_small_terminal_discards_old_mouse_targets` |
| `test_pipeline_bits_are_clipped_and_maximize_reveals_full_value` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_pipeline_bits_are_clipped_and_maximize_reveals_full_value` |
| `test_maximize_switches_panes_and_restores_splits` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_maximize_switches_panes_and_restores_splits` |
| `test_breakpoint_marker_and_run_to_cursor` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_breakpoint_marker_and_run_to_cursor` |
| `test_scalar_edit_refreshes_symbolic_xmem_without_rereads_between_frames` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_scalar_edit_refreshes_symbolic_xmem_without_rereads_between_frames` |
| `test_scalar_editor_rejects_constants_and_invalid_input` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_scalar_editor_rejects_constants_and_invalid_input` |
| `test_scalar_editor_uses_mask_and_cancel_does_not_write` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_scalar_editor_uses_mask_and_cancel_does_not_write` |
| `test_f6_is_unbound_and_cannot_leave_the_tui` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_f6_is_unbound_and_cannot_leave_the_tui` |
| `test_keyboard_interrupt_cancels_from_blocking_or_queued_input` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDebuggerImprovements::test_keyboard_interrupt_cancels_from_blocking_or_queued_input` |
| `test_compact_navigation_keeps_every_pane_accessible` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_compact_navigation_keeps_every_pane_accessible` |
| `test_shrink_and_grow_restores_user_split_sizes` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_shrink_and_grow_restores_user_split_sizes` |
| `test_idle_poll_detects_pty_resize_without_key_resize` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_idle_poll_repaints_after_a_resize_without_key_resize` (renamed) |
| `test_unchanged_terminal_does_not_repaint_on_timeout` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_unchanged_terminal_does_not_repaint_on_timeout` |
| `test_footer_click_steps_and_tab_buttons_have_their_own_direction` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_footer_click_steps_and_tab_buttons_have_their_own_direction` |
| `test_help_scrolls_and_closes_without_clicking_background` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_help_scrolls_and_closes_without_clicking_background` |
| `test_editor_mouse_places_cursor_and_applies_or_cancels` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_editor_mouse_places_cursor_and_applies_or_cancels` |
| `test_scrollbar_drag_reaches_end_and_releases` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_scrollbar_drag_reaches_end_and_releases` |
| `test_wheel_over_tabs_selects_tabs_and_shift_wheel_moves_columns` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_wheel_over_tabs_selects_tabs_and_shift_wheel_moves_columns` |
| `test_code_gutter_toggles_breakpoint_but_instruction_click_does_not` | `src/tools/ipu-debug/test/test_debug_tui.py::TestResponsiveMouseUI::test_code_gutter_toggles_breakpoint_but_instruction_click_does_not` |
| `test_f11_removed_and_footer_help_clickable` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_f11_removed_and_footer_help_clickable` |
| `test_long_tab_fits_smallest_pane_and_can_be_closed` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_long_tab_fits_smallest_pane_and_can_be_closed` |
| `test_grabbing_thumb_does_not_move_selection` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_grabbing_thumb_does_not_move_selection` |
| `test_scroll_drag_can_return_to_its_origin` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_scroll_drag_can_return_to_its_origin` |
| `test_double_click_register_opens_editor_but_slow_clicks_select` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_double_click_register_opens_editor_but_slow_clicks_select` |
| `test_terminal_reported_double_click_edits_selected_register` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_terminal_reported_double_click_edits_selected_register` |
| `test_compact_help_keeps_scroll_range_visible` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_compact_help_keeps_scroll_range_visible` |
| `test_click_only_terminal_preserves_selection_on_thumb` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_click_only_terminal_preserves_selection_on_thumb` |
| `test_compact_help_does_not_waste_space_on_column_padding` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMouseUsability::test_compact_help_does_not_waste_space_on_column_padding` |
| `test_stage_decoding_includes_each_metadata_slot` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_stage_decoding_includes_each_metadata_slot` |
| `test_trace_tracks_actual_branch_pc_and_does_not_record_stops` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_trace_tracks_actual_branch_pc_and_does_not_record_stops` |
| `test_trace_follows_pc_edits_and_resets_for_a_new_run` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_trace_follows_pc_edits_and_resets_for_a_new_run` |
| `test_stages_is_disassembly_format_and_other_panes_remain_usable` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_stages_is_disassembly_format_and_other_panes_remain_usable` |
| `test_stage_view_mouse_scroll_and_execution` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_stage_view_mouse_scroll_and_execution` |
| `test_stage_view_shows_current_commands_grouped_by_physical_stage` | `src/tools/ipu-debug/test/test_tui_disassembly_pane.py::TestStageView::test_stages_group_the_current_commands_by_physical_stage` (renamed) |
| `test_disassembly_shows_slot_names_and_explicit_truncation` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_disassembly_shows_slot_names_and_explicit_truncation` |
| `test_display_uses_semantic_operand_order_and_omits_unused_fields` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_display_uses_semantic_operand_order_and_omits_unused_fields` |
| `test_concurrent_commands_occupy_separate_stages_and_nops_clear_them` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_concurrent_commands_occupy_separate_stages_and_nops_clear_them` |
| `test_stage_occupancy_follows_branch_destination` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPipelineStageView::test_stage_occupancy_follows_branch_destination` |
| `test_removed_hints_help_button_and_group_order` | `src/tools/ipu-debug/test/test_debug_tui.py::TestConsoleBars::test_removed_hints_help_button_and_group_order` |
| `test_positions_shortcuts_and_footer_have_no_focus_label` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPaneOrder::test_positions_shortcuts_and_footer_have_no_focus_label` |
| `test_pipeline_edit_uses_correct_register_backing_and_preserves_neighbors` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_the_value_editor_writes_the_right_storage_and_keeps_its_neighbours` (renamed) |
| `test_xmem_edit_resolves_symbolic_address_and_invalidates_cached_views` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMemoryValueEditing::test_xmem_edit_resolves_symbolic_address_and_invalidates_cached_views` |
| `test_double_click_invalid_input_and_cancel_leave_bytes_unchanged` | `src/tools/ipu-debug/test/test_debug_tui.py::TestMemoryValueEditing::test_double_click_invalid_input_and_cancel_leave_bytes_unchanged` |
| `test_terminal_sets_short_escape_sequence_timeout` | `src/tools/ipu-debug/test/test_tui_shell.py::TestCursesScreen::test_terminal_sets_short_escape_sequence_timeout` |
| `test_q_does_not_close_help_or_quit` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPopupEscape::test_q_does_not_close_help_or_quit` |
| `test_escape_cancels_each_value_popup` | `src/tools/ipu-debug/test/test_debug_tui.py::TestPopupEscape::test_escape_cancels_each_value_popup` |
| `test_arrow_moves_divider_in_screen_direction` | `src/tools/ipu-debug/test/test_debug_tui.py::TestStatusAndDividerDirection::test_arrow_moves_divider_in_screen_direction` |
| `test_status_is_left_of_help_and_footer_stays_shortcuts` | `src/tools/ipu-debug/test/test_debug_tui.py::TestStatusAndDividerDirection::test_status_is_left_of_help_and_footer_stays_shortcuts` |
| `test_close_button_accepts_terminal_mouse_events` | `src/tools/ipu-debug/test/test_debug_tui.py::TestHelpScrollbar::test_close_button_accepts_terminal_mouse_events` |
| `test_help_scrollbar_pages_drags_and_releases_without_touching_panes` | `src/tools/ipu-debug/test/test_debug_tui.py::TestHelpScrollbar::test_help_scrollbar_pages_drags_and_releases_without_touching_panes` |
| `test_fresh_tabs_cover_all_registers_and_use_storage_formats` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDefaultWorkspace::test_fresh_tabs_cover_all_registers_and_use_storage_formats` |
| `test_default_sizes_balance_rows_and_manual_sizes_survive_resize` | `src/tools/ipu-debug/test/test_debug_tui.py::TestDefaultWorkspace::test_default_sizes_balance_rows_and_manual_sizes_survive_resize` |
| `test_slots_have_separators_and_preserve_cursor_byte_mapping` | `src/tools/ipu-debug/test/test_tui_pipeline_pane.py::test_mask_slots_have_separators_and_keep_values_inside_one_slot` (renamed) |

### Tests from `src/tools/ipu-as-py/test/test_gen_codegen.py`

All 13 tests moved with the file. None was dropped.

| Test | Now covered by |
|---|---|
| `test_build_context_matches_compound_width` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_build_context_width_is_the_hardware_word` (renamed in E03-22: the SV word is now the HARDWARE word, 155 bits; the test also pins the emulator word, 186 bits) |
| `test_slot_union_struct_bit_widths` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_slot_union_struct_bit_widths` |
| `test_compound_members_follow_slot_order` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_compound_members_follow_slot_order` |
| `test_generate_sv_package_is_proper_systemverilog` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_generate_sv_package_is_proper_systemverilog` |
| `test_union_members_padded_to_operand_width` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_union_members_padded_to_operand_width` |
| `test_union_member_fields_named_after_their_operands` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_union_member_fields_named_after_their_operands` |
| `test_unused_union_columns_named_padding` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_unused_union_columns_named_padding` |
| `test_union_members_exclude_opcode` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_union_members_exclude_opcode` |
| `test_nop_union_members_use_single_padding_field` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_nop_union_members_use_single_padding_field` |
| `test_render_is_deterministic` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_render_is_deterministic` |
| `test_header_names_the_hash_of_the_body_not_a_commit` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_header_names_the_hash_of_the_body_not_a_commit` |
| `test_verify_accepts_the_package_and_rejects_any_edit` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_verify_accepts_the_package_and_rejects_any_edit` |
| `test_compound_comment_lists_the_layout_order` | `src/tools/ipu-gen/test/test_gen_codegen.py::test_compound_comment_lists_the_layout_order` |

### Tests from `src/tools/ipu-as-py/test/test_isa_data.py`

All 7 tests moved with the file, under the same names. None was dropped.

| Test | Now covered by |
|---|---|
| `test_every_operand_type_is_described` | `src/tools/ipu-gen/test/test_isa_data.py::test_every_operand_type_is_described` |
| `test_every_offered_value_is_accepted` | `src/tools/ipu-gen/test/test_isa_data.py::test_every_offered_value_is_accepted` |
| `test_range_is_exact` | `src/tools/ipu-gen/test/test_isa_data.py::test_range_is_exact` |
| `test_withheld_enum_values_are_the_undecodable_ones` | `src/tools/ipu-gen/test/test_isa_data.py::test_withheld_enum_values_are_the_undecodable_ones` |
| `test_every_instruction_assembles_from_completion_values` | `src/tools/ipu-gen/test/test_isa_data.py::test_every_instruction_assembles_from_completion_values` |
| `test_lexical_facts_match_the_grammar_and_the_assembler` | `src/tools/ipu-gen/test/test_isa_data.py::test_lexical_facts_match_the_grammar_and_the_assembler` |
| `test_every_register_is_described_by_its_definition` | `src/tools/ipu-gen/test/test_isa_data.py::test_every_register_is_described_by_its_definition` |

## E07 Kernel migration

Rows are added by each E07 issue as it deletes the old ipu-apps kernels.

| Deleted | Issue | Why | Replaced by | Evidence |
|---|---|---|---|---|
| `src/tools/ipu-apps/src/ipu_apps/kernels/projections/` | E07-02 (e718ae69) | The 12 projection kernels moved onto `ipu_harness`: per-shape folders (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) and the family's `app.py`, `cases.py`, `test.py`. | `kernels/projections/`: 12 `registry.toml` entries over the templates `proj_p4` (8), `proj_p4_2tg` (3) and `proj_p4_2tg_silu` (1), plus the family's `pack.py`, `gen.py` and `test_routing.py`. Targets keep their names under `//kernels/projections:`. | `compare_setup`: IDENTICAL for all 12 (CR0-CR15, LR0-LR15, XMEM). `compare_legacy projections`: 12/12 cases identical in binary, output and cycles (for example `proj_qkv_240_p4` 748832/748832). Every `:assemble_<name>.bin` matches its E02 digest. The MobileViT L4/L5 full-layer tests pass through the temporary adapter `test/mobilevit/fixture_projections.py` (7/7 each). |
| `src/tools/ipu-kernels/src/ipu_kernels/reference/ops.py` | E07 gap 10 (lane B) | One module per op, so each family's migration adds its op's file without touching the others'. | `src/tools/ipu-kernels/src/ipu_kernels/reference/ops/`: `matmul.py`, `linear.py`, `silu.py`, `softmax.py`, `layernorm.py` (functions moved unchanged), re-exported by `ops/__init__.py` under the same names, so `op = "<name>"` resolves as before. | `//src/tools/ipu-kernels:test_reference_ops` passes unchanged, plus `test_each_op_is_its_own_module_named_for_it`; `//kernels/projections/...` (op `linear`) passes. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/concat_8x8x160x160/` | E07-04 | The concat moved onto `ipu_harness`; its three shape copies had one body. The folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes, and with it the family `app.py`'s concat query and spec and `cases.py`'s `concat_cases` (deleted lines, not files). | The entry `concat_8x8x160x160` of `kernels/reshape/registry.toml` over the template `kernels/reshape/concat/` (`//kernels/reshape:concat_8x8x160x160`); shared reference `ops.concat` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/concat.py`). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: default case identical in binary, output and cycles (973/973). E02 asm digest `b762282246d3725f` (all three), equal to `:assemble_concat_8x8x160x160`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/concat_16x16x128x128/` | E07-04 | The concat moved onto `ipu_harness`; its three shape copies had one body. The folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes, and with it the family `app.py`'s concat query and spec and `cases.py`'s `concat_cases` (deleted lines, not files). | The entry `concat_16x16x128x128` of `kernels/reshape/registry.toml` over the template `kernels/reshape/concat/` (`//kernels/reshape:concat_16x16x128x128`); shared reference `ops.concat` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/concat.py`). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: default case identical in binary, output and cycles (781/781). E02 asm digest `b762282246d3725f` (all three), equal to `:assemble_concat_16x16x128x128`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/concat_32x32x96x96/` | E07-04 | The concat moved onto `ipu_harness`; its three shape copies had one body. The folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes, and with it the family `app.py`'s concat query and spec and `cases.py`'s `concat_cases` (deleted lines, not files). | The entry `concat_32x32x96x96` of `kernels/reshape/registry.toml` over the template `kernels/reshape/concat/` (`//kernels/reshape:concat_32x32x96x96`); shared reference `ops.concat` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/concat.py`). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: default case identical in binary, output and cycles (589/589). E02 asm digest `b762282246d3725f` (all three), equal to `:assemble_concat_32x32x96x96`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/fold_8x8x240/` | E07-05 | The fold kernel moved onto `ipu_harness`; its folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes. | `kernels/reshape/fold_8x8x240/` (`//kernels/reshape/fold_8x8x240`); shared reference `ops.fold` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/fold.py`); layouts from the family `pack.py` (`phase_rows` for x, `stripes` for y). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: identical in binary, output and cycles (7683/7683). E02 asm digest `ad385d9c50df76b1`, equal to `:assemble_fold_8x8x240`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/fold_16x16x192/` | E07-05 | The fold kernel moved onto `ipu_harness`; its folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes. | `kernels/reshape/fold_16x16x192/` (`//kernels/reshape/fold_16x16x192`); shared reference `ops.fold` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/fold.py`); layouts from the family `pack.py` (`phase_rows` for x, `stripes` for y). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: identical in binary, output and cycles (17859/17859). E02 asm digest `3ca5372afadf1052`, equal to `:assemble_fold_16x16x192`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/fold_32x32x144/` | E07-05 | The fold kernel moved onto `ipu_harness`; its folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes. | `kernels/reshape/fold_32x32x144/` (`//kernels/reshape/fold_32x32x144`); shared reference `ops.fold` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/fold.py`); layouts from the family `pack.py` (`phase_rows` for x, `stripes` for y). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: identical in binary, output and cycles (58776/58776). E02 asm digest `5a31445bbb08dee0`, equal to `:assemble_fold_32x32x144`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/unfold_8x8x240/` | E07-05 | The unfold kernel moved onto `ipu_harness`; its folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes. | `kernels/reshape/unfold_8x8x240/` (`//kernels/reshape/unfold_8x8x240`); shared reference `ops.unfold` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/unfold.py`); layouts from the family `pack.py` (`paired_rows` for x, `phase_rows` for y). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: identical in binary, output and cycles (2883/2883 (and `garbage_padding` 2883/2883)). E02 asm digest `08dc2c8aff74113e`, equal to `:assemble_unfold_8x8x240`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/unfold_16x16x192/` | E07-05 | The unfold kernel moved onto `ipu_harness`; its folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes. | `kernels/reshape/unfold_16x16x192/` (`//kernels/reshape/unfold_16x16x192`); shared reference `ops.unfold` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/unfold.py`); layouts from the family `pack.py` (`stripes` for x, `phase_rows` for y). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: identical in binary, output and cycles (3075/3075). E02 asm digest `24b18a9a145c8dd6`, equal to `:assemble_unfold_16x16x192`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/unfold_32x32x144/` | E07-05 | The unfold kernel moved onto `ipu_harness`; its folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`, the `.asm`) goes. | `kernels/reshape/unfold_32x32x144/` (`//kernels/reshape/unfold_32x32x144`); shared reference `ops.unfold` (`src/tools/ipu-kernels/src/ipu_kernels/reference/ops/unfold.py`); layouts from the family `pack.py` (`stripes` for x, `phase_rows` for y). | `compare_setup`: IDENTICAL (CR0-CR15, LR0-LR15, XMEM). `compare_legacy reshape`: identical in binary, output and cycles (6339/6339). E02 asm digest `e60c1094cf54db21`, equal to `:assemble_unfold_32x32x144`. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/app.py` | E07-05 | The family's unfold/fold query vocabulary and `SPEC` helpers (`unfold_query`, `unfold_spec`, `positive_dims`); every kernel that used them moved. | The kernels' `[params]` (`h`, `w`, `c`, one choice each) and `kernels/reshape/test_routing.py`. | No user left: `git grep "kernels.reshape.app"` finds nothing. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/cases.py` | E07-05 | The family's shared cases (`spatial_tensor`, `stripe_input`, `stream_rows`, `unfold_cases`, `fold_cases`, the checks). | `kernels/reshape/gen.py` (`spatial`, `phases`: the same seeds and values), the pack transforms in `kernels/reshape/pack.py`, and `ops.unfold`/`ops.fold` as the references. | compare_setup/compare_legacy above: the new cases' inputs are byte-identical to the old ones in XMEM. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/test.py` | E07-05 | The family test (fold inverts unfold, plain and with R_ACC poisoned; shared shapes). | `kernels/reshape/test_fold_unfold.py` (`//kernels/reshape:test_fold_unfold`), on `ipu_harness.chain`. | Tests mapped below. |

Kept until a later issue (the old folder stays frozen, logged here when it goes):

- `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/identity/`: kept until E07-25. E07-04 migrated it (`kernels/reshape/identity`, compare_setup and compare_legacy identical: 17/17 and 7/7 cycles, E02 digest `65e08a2907637a18`), but the old folder is the generic fixture of the old harness's own tests: `src/tools/ipu-apps/test/{test_harness_factory,test_debug_launch,test_bazel_entry,test_kernel_registry,test_kernel_baseline,test_reference_model_kernels,test_asm_digests}.py`, `src/tools/ipu-apps/BUILD.bazel` (data globs), `kernels/tools/test_compare.py`, the Inspector trace `docs/inspector` (`identity`), `docs/test_doc_commands.py`, `tools/bench` (the `kernel/identity/*` series) and the `getting-started.tex` tutorial. E07-25 deletes them with `ipu-apps`.
- `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/depth_to_space/`: kept until E07-26. E07-04 migrated it (`kernels/reshape/depth_to_space`, identical in all 4 cases, E02 digest `cd1e0e2be66c1bf8`), but the Inspector's curated `depth_to_space` trace is recorded through `//src/tools/ipu-apps:trace_case` (`docs/inspector.bzl`), which E07-26 moves onto the runner.
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_rows_long/` | E07-11 (735152d2) | The kernel moved onto `ipu_harness`: `app.py` (`SoftmaxRowsLongApp`: the chunk geometry, CR8-CR14, `SPEC`), `cases.py`, `test.py` (16-point sweep), `benchmark.py`, `results.md`, `baseline.json`, `__init__.py`, and the `.asm` (its header rewritten, so git sees no rename). | `kernels/softmax/softmax_rows_long/`: `softmax_rows_long.asm` (CRs by `[abi]` name, upper case) and `kernel.toml` (`[derived]`, `[abi]`, `cost = 1`, cases `default`, `layout_tail`, `layout_exact` and `sweep_01`..`sweep_16`, `[benchmark]`). `//kernels/softmax/softmax_rows_long` keeps the target name. | `compare_setup` IDENTICAL for all 17 old cases (CR0-CR15, LR0-LR15, XMEM). `compare_legacy softmax`: 17/17 binary and output equal, cycles equal (`default` 570/570). E02 asm `bin_sha256` b6f6c0bfab75…, the default program's. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_columns/` | E07-11 (735152d2) | The kernel moved onto `ipu_harness`: `app.py` (`SoftmaxColumnsApp`: chunk columns, the cpr-row max and rvec regions, CR9-CR13, `SPEC` with the idle-lanes caveat), `cases.py`, `test.py` (16-point sweep), `benchmark.py`, `results.md`, `baseline.json`, `__init__.py`, and the `.asm` (its header rewritten, so git sees no rename). | `kernels/softmax/softmax_columns/`: `softmax_columns.asm` and `kernel.toml` (cases `default`, `layout_pad`, `layout_exact`, `sweep_01`..`sweep_16`, `[benchmark]`). The idle-lanes caveat is in the `kernel.toml` and `.asm` headers (a `Verdict` has no caveats). | `compare_setup` IDENTICAL for all 17 old cases. `compare_legacy softmax`: 17/17 equal (`default` 1239/1239). E02 asm `bin_sha256` 251f37b7eebc…. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/test.py` | E07-11 (735152d2) | Family tests: the layout contract, the router, torch's `dim` convention, the docs routing tables. | `kernels/softmax/test_routing.py`, `test_layers.py`, `test_pack.py` and the kernels' `layout*` cases (`//kernels/softmax:tests`); every test is mapped below. | 28 test functions: 23 have a new home, 5 are dropped with their reason (below). |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_rows/benchmark.py` | E07-11 (735152d2) | `CONFIGS`/`PER` of `:benchmark_softmax_rows`. The folder stays as a temporary shim (below); this file moved. | `[benchmark]` in `kernels/softmax/softmax_rows/kernel.toml` (points = `CONFIGS`, `per = "rows"`): `bazel run //kernels/softmax/softmax_rows -- --benchmark`. | The new table's cycles equal the old `results.md`'s at every point (rows 8..500: 162, 594, 2323, 4643, 9066). |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_rows/results.md` | E07-11 (735152d2) | Hand-pasted benchmark output. The folder stays as a temporary shim (below); this file moved. | The `--benchmark` output; the steady-state cost (about 18 cycles per row) is in the `.asm` header. | As above. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_rows_partial/benchmark.py` | E07-11 (735152d2) | `CONFIGS`/`PER` of `:benchmark_softmax_rows_partial`. The folder stays as a temporary shim (below); this file moved. | `[benchmark]` in `kernels/softmax/softmax_rows_partial/kernel.toml`. | The new table's cycles equal the old `results.md`'s (for example n=16, 512 rows: 9435). |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_rows_partial/results.md` | E07-11 (735152d2) | Hand-pasted benchmark output. The folder stays as a temporary shim (below); this file moved. | The `--benchmark` output. | As above. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_rows_partial/STATUS.md` | E07-11 (735152d2) | Design and bug-history notes (the Pass-4 cross-partition masks, the row-count overflow and the group loop, open work). The folder stays as a temporary shim (below); this file moved. | The `.asm` header of `kernels/softmax/softmax_rows_partial/` (GROUP LOOP, PASS-4 MASKS, Open), Q79. Its stale points (the old fixed NUM address, byte-scaled constants) are not carried over. | Read side by side; the two shim files that cited it now cite the `.asm` header. |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_columns_packed/benchmark.py` | E07-11 (735152d2) | `CONFIGS`/`PER` of `:benchmark_softmax_columns_packed`. The folder stays as a temporary shim (below); this file moved. | `[benchmark]` in `kernels/softmax/softmax_columns_packed/kernel.toml`. | The new table's cycles equal the old `results.md`'s (width 32, 100 rows: 525). |
| `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_columns_packed/results.md` | E07-11 (735152d2) | Hand-pasted benchmark output. The folder stays as a temporary shim (below); this file moved. | The `--benchmark` output. | As above. |
| `src/tools/ipu-apps/test/baselines/disassembly/kernels/softmax/softmax_columns/softmax_columns.txt` | E07-11 (735152d2) | The disassembly snapshot of a deleted `.asm` (`:update_disassembly_snapshots` drops it). | — (the program is `kernels/softmax/softmax_columns/softmax_columns.asm`, pinned by its E02 digest through `compare_legacy`) | Deletion only: `tools/update_baselines.sh` changed nothing else. |
| `src/tools/ipu-apps/test/baselines/disassembly/kernels/softmax/softmax_rows_long/softmax_rows_long.txt` | E07-11 (735152d2) | As above. | As above (`kernels/softmax/softmax_rows_long/softmax_rows_long.asm`). | As above. |

**Kept until later issues (the softmax shim).** `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/` keeps `__init__.py`, `app.py` (`SoftmaxApp`, `softmax_spec`, and the `Softmax`/`LogSoftmax`/`Softmin` layer adapters of the old registry), `cases.py` (`run_array`), and the folders `softmax_rows/`, `softmax_rows_partial/` and `softmax_columns_packed/` (`app.py`, `cases.py`, `test.py`, `baseline.json`, the `.asm`). They are kept until E07-12 (`attention/test.py` imports `run_array`, `SoftmaxRowsPartialApp` and `SoftmaxColumnsPackedApp`), E07-24 (the MobileViT L4/L5 tests and `fixture_full_layer_chain.py` import the two apps) and E07-26 (the Cycle Inspector's `softmax_rows` trace and `test_harness_factory.py` run `softmax_rows` through `//src/tools/ipu-apps:trace_case` and `load_cases`); the issue that removes the last user deletes them, with rows here. The new kernels are the tested ones; the shim's programs are the old ones, byte for byte (`compare_legacy`). When the shim goes, its `softmax_rows_partial/test.py` takes with it `test_scatter_counts_every_partition`'s `mult_identity_cycles == 2 * rows` assertion, which the reference model's statistics cannot express (the new test keeps the lane counts); that issue's row records it.
| `src/tools/ipu-apps/src/ipu_apps/kernels/conformance/` | E07-26 | `vector_tour`, the Cycle Inspector's conformance program, moved onto `ipu_harness`: its folder (`app.py`, `cases.py`, `test.py`, `__init__.py`, `baseline.json`; the `.asm` is renamed) and the family's `__init__.py`. | `kernels/conformance/vector_tour/`: `vector_tour.asm`, `kernel.toml` (cases `default`, `int8`, `e4`, `e5`, `e3`, one format each), `gen.py` (the old `default_rng(16)` data and the R_MASK bytes) and `reference.py`. The `tour_*` Inspector traces are recorded through `run_kernel --trace --context`. | `compare_setup`: IDENTICAL for all 5 cases (CR0-CR15, LR0-LR15, 896 XMEM bytes). `compare_legacy conformance`: 5/5 cases identical in binary, output and cycles (INT8 35/35, E4/E5/E3 32/32). The four `tour_*` `trace.jsonl` files are byte-equal; their page JSON differs only in `source_path` and `source`. |
| `src/tools/ipu-apps/test/baselines/disassembly/kernels/conformance/vector_tour/vector_tour.txt` | E07-26 | The disassembly snapshot of the deleted `ipu_apps` kernel; `update_disassembly_snapshots` covers `ipu_apps` kernels only. | `//kernels/conformance/vector_tour:assemble_vector_tour` (the program's `.bin`); the binary is the E02 asm digest (`compare_legacy`). | Removed by `tools/update_baselines.sh`, with its `asm_digests.json` entry (deletions only). |

### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/projections/<kernel>/test.py` (11 files)

| Test | Now covered by |
|---|---|
| `test_case` (`case_tests(__package__)`: every case in the folder's `cases.py`) | The entry's generated tests, `bazel test //kernels/projections:<kernel>` (the default case and its formats); the old data is the default case (`gen.py:legacy_uniform`). |

### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/projections/test.py`

| Test | Now covered by |
|---|---|
| `test_the_family_has_all_twelve_kernels` | `kernels/projections/test_routing.py::test_the_family_is_twelve_entries_over_three_templates` |
| `test_each_kernel_is_routed_by_its_exact_query` | `kernels/projections/test_routing.py::test_each_kernel_is_routed_by_its_exact_query`, and `test_the_query_tool_gives_the_same_answer` through the query tool |
| `test_the_wrong_activation_is_refused` | `kernels/projections/test_routing.py::test_the_wrong_activation_is_refused` |
| `test_ffn1_kernels_are_exactly_the_silu_kernels` | `kernels/projections/test_routing.py::test_ffn1_kernels_are_exactly_the_silu_kernels` |
| `test_other_stream_counts_and_unknown_activations_are_refused` | `kernels/projections/test_routing.py::test_other_stream_counts_and_unknown_activations_are_refused` |
| `test_the_family_base_harness_cannot_be_constructed` | dropped: it tested the legacy `ipu_apps` base-harness class, which no longer exists; a template is not routed (`Registry.templates`). |
| `test_spec_activation_matches_the_asm_store` | dropped: the activation is now rendered into the program from the entry's `activation` parameter, so the spec and the store cannot disagree; the `:assemble_<name>` digests pin each store word. |
| `test_proj_p4_weight_row_is_not_stuck_at_channel_zero` | `kernels/projections/proj_p4/kernel.toml` nightly case `weight_rows` (`gen.py:weight_rows`), run by `//kernels/projections/proj_p4:proj_p4_nightly`. |

### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/concat_<shape>/test.py` (3 files)

| Test | Now covered by |
|---|---|
| `test_case` (`case_tests(__package__)`: the `default` case of `concat_cases`) | The entry's generated tests, `bazel test //kernels/reshape:concat_<shape>`: the template's `default` case at the entry's shape, the old data (`gen.py:legacy_uniform`, `RandomState(0xC047)`) held EXACT to `ops.concat`. |
### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/test.py`

28 test functions. The family tests run as `//kernels/softmax:tests` against the kernel tree's registry (`Registry.resolve`, `Registry.from_layer`). A legacy verdict "constructs" its app; a harness verdict has no constructor, so "constructs" becomes "plans, places and sets its registers" (`plan_case`, `place`, `register_writes`). Five functions are dropped, each with its reason; three keep their routing half and lose a caveat or note half, because an `ipu_harness` `Verdict` carries no caveats or shape notes.

| Test | Now covered by |
|---|---|
| `test_output_file_matches_input_layout` (10 ids: `rows`, `partial_pad`, `partial_exact`, `partial_p8`, `long_tail`, `long_exact`, `columns_pad`, `columns_exact`, `cols_packed_pad`, `cols_packed_exact`) | One kernel case per id, at the same shape and data (`seed = rows + cols`, scale 3.0, `../gen.py:legacy_randn`): `softmax_rows` `layout`; `softmax_rows_partial` `layout_pad`, `layout_exact`, `layout_p8`; `softmax_rows_long` `layout_tail`, `layout_exact`; `softmax_columns` `layout_pad`, `layout_exact`; `softmax_columns_packed` `layout_pad`, `layout_exact`. The runner unpacks `y` through its declared `pack` to `(rows, cols)` and holds it to the reference, which pins element order; `kernels/softmax/test_pack.py::test_rows_in_slots_round_trip_with_the_padding_filled` pins the slot layouts and their padding. |
| `test_supported_rows_verdicts_construct` (75 ids) | `kernels/softmax/test_routing.py::test_every_row_query_is_accepted_and_runs` (rows 1, 4, 128, 129, 200, 1000 over the same 15 lengths; it also asserts each query is accepted) |
| `test_supported_column_verdicts_construct` (33 ids) | `kernels/softmax/test_routing.py::test_every_column_query_is_accepted_and_runs` (the same grid) |
| `test_rows_axis_routing` (6 ids) | `kernels/softmax/test_routing.py::test_rows_axis_routing` (the same 6) |
| `test_columns_axis_routing` (4 ids) | `kernels/softmax/test_routing.py::test_columns_axis_routing` (the same 4, and 65, 96, 127) |
| `test_no_verdict_warns_about_packed_output` | dropped: a harness `Verdict` has no caveats, so none can warn. The property (the output's layout is the input's) is each output's `pack` declaration, held by the `layout*` cases. |
| `test_rows_multiple_of_128_supported` (4 ids) | `kernels/softmax/test_routing.py::test_a_multiple_of_128_is_a_long_row_with_no_tail_chunk` (the plan's `tail == 0`, `cpr == n // 128`); the runs are `softmax_rows_long` cases `sweep_09`..`sweep_12` (n = 256, 384, 512, 1024) |
| `test_column_width_65_to_127_supported` (3 ids) | `kernels/softmax/test_routing.py::test_columns_axis_routing[65-…]`, `[96-…]`, `[127-…]`; the runs are `softmax_columns` cases `sweep_13`..`sweep_15`. The "idle lanes" caveat half is dropped (no caveats); the `softmax_columns` `kernel.toml` and `.asm` headers state it. |
| `test_no_open_gaps` (2 ids) | `kernels/softmax/test_routing.py::test_no_open_gaps` (`ipu_harness.query.boundaries`, lengths 1..599) |
| `test_rows_over_128_supported` (12 ids) | `kernels/softmax/test_routing.py::test_every_row_query_is_accepted_and_runs` (rows 129, 200, 1000 at n 8, 32, 128, 300 among others). The "no overflow caveat" half is dropped (no caveats). |
| `test_columns_axis_has_no_row_cap` | `kernels/softmax/test_routing.py::test_the_columns_axis_has_no_row_cap` |
| `test_verdict_is_truthy_only_when_supported` | dropped: the legacy `Verdict` API. `ipu_harness`'s is covered by `src/tools/ipu-harness/test/test_registry.py::test_a_query_no_kernel_takes_is_a_refusal_naming_each_reason`. |
| `test_nonpositive_sizes_unsupported` (2 ids) | `kernels/softmax/test_routing.py::test_nonpositive_sizes_are_refused` (every size parameter has `min = 1`) |
| `test_describe_reports_both_outcomes` | dropped: the legacy `Verdict.describe` text. The query tool's text is covered by `src/tools/ipu-harness/test/test_registry.py::test_query_answers_a_query_a_sweep_and_the_report`. |
| `test_torch_negative_dims_count_from_the_end` (2 ids) | `kernels/softmax/test_layers.py::test_a_negative_dim_counts_from_the_end` |
| `test_torch_1d_shape_is_a_single_row` (3 ids) | `kernels/softmax/test_layers.py::test_a_1d_tensor_is_one_row` |
| `test_torch_verdicts_construct` | `kernels/softmax/test_layers.py::test_a_2d_tensor_routes_by_its_axis_and_plans` (the same 4 shapes) |
| `test_higher_rank_is_flattened_and_disclosed` (2 ids) | `kernels/softmax/test_layers.py::test_a_higher_rank_tensor_is_a_batch_of_2d_problems` (the same 2). The disclosure half is dropped: a harness `Verdict` carries no shape notes; `ipu_kernels/softmax/layers.py` documents the flattening. |
| `test_interior_reduction_axis_is_refused` | `kernels/softmax/test_layers.py::test_an_interior_axis_is_refused` |
| `test_torch_rejects_out_of_range_dim` (3 ids) | `kernels/softmax/test_layers.py::test_an_out_of_range_dim_is_refused` (the same 3; `UnsupportedLayer` is a `ValueError`) |
| `test_torch_dim_choice_actually_changes_the_answer` | `kernels/softmax/test_routing.py::test_the_axis_decides_the_kernel` and `kernels/softmax/test_layers.py::test_the_axis_decides_the_kernel` |
| `test_torch_agrees_with_real_pytorch_semantics` | `kernels/softmax/test_layers.py::test_the_routed_axis_is_the_one_torch_normalises` (the same 6 shapes; torch is a test dependency, so no skip) |
| `test_case_width_must_be_declared` | dropped: `random_case` is gone. A case names its values in `kernel.toml`, and a misspelled key is a load error (`src/tools/ipu-harness/test/test_manifest.py`). |
| `test_constructors_require_their_sizes` (4 ids) | dropped: there are no constructors. `rows`, `n` and `width` have no default, so a query that leaves one out is refused by `check_params`, and one that names the other axis's length is refused as an unknown parameter (`src/tools/ipu-harness/test/test_params.py`). |
| `test_input_file_of_the_wrong_size_is_refused_by_name` | `src/tools/ipu-harness/test/test_cases.py::test_a_case_file_of_the_wrong_shape_dtype_or_values_is_rejected` (exact input sizes, L15); an `--input` override goes through the same check. |
| `test_the_doc_actually_contains_routing_tables` | `kernels/softmax/test_routing.py::test_the_doc_has_routing_tables` (`//docs:doc_sources` is data of `:tests`) |
| `test_documented_routing_matches_the_registry` (5 ids) | `kernels/softmax/test_routing.py::test_documented_routing_matches_the_registry` (the same 5) |
| `test_documented_boundaries_are_real_boundaries` (5 ids) | `kernels/softmax/test_routing.py::test_documented_boundaries_are_real_boundaries` (the same 5) |

### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_rows_long/test.py`

One function, `test_case` (`case_tests(__package__, sweep=[...])`): 17 ids, `default` and 16 sweep points, each ported as its own case with the old seed and the old comment. A legacy id ran the case and its check (max error < 1e-4, sums 1 ± 1e-5) and compared it with `baseline.json`; the new case is held to its golden at TRANSCENDENT (atol 1e-4 plus rtol 1e-3, a little looser per element; the issue sets the tier), `kernels/softmax/test_conformance.py` keeps the 1e-4 and sums checks over the registry's choices, and `compare_legacy` proved each case's output bit-identical against `baseline.json` once.

| Test | Now covered by |
|---|---|
| `test_case[softmax_rows_long-default]` | `//kernels/softmax/softmax_rows_long` `test_case[default-fp32]`; `compare_legacy` row `softmax_rows_long` / `default` |
| `test_case[softmax_rows_long-rows=1,n=129,scale=3.0,seed=0]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_01[rows=1,n=129,scale=3.0,seed=0]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=1,n=129,scale=3.0,seed=0` |
| `test_case[softmax_rows_long-rows=4,n=200,scale=4.0,seed=1]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_02[rows=4,n=200,scale=4.0,seed=1]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=4,n=200,scale=4.0,seed=1` |
| `test_case[softmax_rows_long-rows=8,n=257,scale=3.0,seed=2]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_03[rows=8,n=257,scale=3.0,seed=2]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=8,n=257,scale=3.0,seed=2` |
| `test_case[softmax_rows_long-rows=8,n=300,scale=5.0,seed=3]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_04[rows=8,n=300,scale=5.0,seed=3]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=8,n=300,scale=5.0,seed=3` |
| `test_case[softmax_rows_long-rows=16,n=130,scale=50.0,seed=4]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_05[rows=16,n=130,scale=50.0,seed=4]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=16,n=130,scale=50.0,seed=4` |
| `test_case[softmax_rows_long-rows=8,n=300,scale=0.01,seed=5]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_06[rows=8,n=300,scale=0.01,seed=5]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=8,n=300,scale=0.01,seed=5` |
| `test_case[softmax_rows_long-rows=32,n=401,scale=3.0,seed=6]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_07[rows=32,n=401,scale=3.0,seed=6]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=32,n=401,scale=3.0,seed=6` |
| `test_case[softmax_rows_long-rows=128,n=129,scale=4.0,seed=7]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_08[rows=128,n=129,scale=4.0,seed=7]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=128,n=129,scale=4.0,seed=7` |
| `test_case[softmax_rows_long-rows=6,n=256,scale=5.0,seed=8]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_09[rows=6,n=256,scale=5.0,seed=8]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=6,n=256,scale=5.0,seed=8` |
| `test_case[softmax_rows_long-rows=6,n=384,scale=5.0,seed=9]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_10[rows=6,n=384,scale=5.0,seed=9]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=6,n=384,scale=5.0,seed=9` |
| `test_case[softmax_rows_long-rows=4,n=512,scale=3.0,seed=10]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_11[rows=4,n=512,scale=3.0,seed=10]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=4,n=512,scale=3.0,seed=10` |
| `test_case[softmax_rows_long-rows=2,n=1024,scale=4.0,seed=11]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_12[rows=2,n=1024,scale=4.0,seed=11]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=2,n=1024,scale=4.0,seed=11` |
| `test_case[softmax_rows_long-rows=129,n=129,scale=3.0,seed=12]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_13[rows=129,n=129,scale=3.0,seed=12]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=129,n=129,scale=3.0,seed=12` |
| `test_case[softmax_rows_long-rows=200,n=200,scale=4.0,seed=13]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_14[rows=200,n=200,scale=4.0,seed=13]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=200,n=200,scale=4.0,seed=13` |
| `test_case[softmax_rows_long-rows=256,n=130,scale=3.0,seed=14]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_15[rows=256,n=130,scale=3.0,seed=14]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=256,n=130,scale=3.0,seed=14` |
| `test_case[softmax_rows_long-rows=300,n=129,scale=5.0,seed=15]` | `//kernels/softmax/softmax_rows_long` `test_case[sweep_16[rows=300,n=129,scale=5.0,seed=15]-fp32]`; `compare_legacy` row `softmax_rows_long` / `rows=300,n=129,scale=5.0,seed=15` |

### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/softmax/softmax_columns/test.py`

One function, `test_case` (`case_tests(__package__, sweep=[...])`): 17 ids, `default` and 16 sweep points, each ported as its own case with the old seed and the old comment. A legacy id ran the case and its check (max error < 1e-4, sums 1 ± 1e-5) and compared it with `baseline.json`; the new case is held to its golden at TRANSCENDENT (atol 1e-4 plus rtol 1e-3, a little looser per element; the issue sets the tier), `kernels/softmax/test_conformance.py` keeps the 1e-4 and sums checks over the registry's choices, and `compare_legacy` proved each case's output bit-identical against `baseline.json` once.

| Test | Now covered by |
|---|---|
| `test_case[softmax_columns-default]` | `//kernels/softmax/softmax_columns` `test_case[default-fp32]`; `compare_legacy` row `softmax_columns` / `default` |
| `test_case[softmax_columns-rows=8,width=128,scale=3.0,seed=0]` | `//kernels/softmax/softmax_columns` `test_case[sweep_01[rows=8,width=128,scale=3.0,seed=0]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=8,width=128,scale=3.0,seed=0` |
| `test_case[softmax_columns-rows=64,width=128,scale=4.0,seed=1]` | `//kernels/softmax/softmax_columns` `test_case[sweep_02[rows=64,width=128,scale=4.0,seed=1]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=64,width=128,scale=4.0,seed=1` |
| `test_case[softmax_columns-rows=1,width=128,scale=5.0,seed=2]` | `//kernels/softmax/softmax_columns` `test_case[sweep_03[rows=1,width=128,scale=5.0,seed=2]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=1,width=128,scale=5.0,seed=2` |
| `test_case[softmax_columns-rows=16,width=130,scale=3.0,seed=3]` | `//kernels/softmax/softmax_columns` `test_case[sweep_04[rows=16,width=130,scale=3.0,seed=3]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=16,width=130,scale=3.0,seed=3` |
| `test_case[softmax_columns-rows=32,width=200,scale=4.0,seed=4]` | `//kernels/softmax/softmax_columns` `test_case[sweep_05[rows=32,width=200,scale=4.0,seed=4]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=32,width=200,scale=4.0,seed=4` |
| `test_case[softmax_columns-rows=128,width=256,scale=3.0,seed=5]` | `//kernels/softmax/softmax_columns` `test_case[sweep_06[rows=128,width=256,scale=3.0,seed=5]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=128,width=256,scale=3.0,seed=5` |
| `test_case[softmax_columns-rows=64,width=192,scale=50.0,seed=6]` | `//kernels/softmax/softmax_columns` `test_case[sweep_07[rows=64,width=192,scale=50.0,seed=6]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=64,width=192,scale=50.0,seed=6` |
| `test_case[softmax_columns-rows=10,width=129,scale=0.01,seed=7]` | `//kernels/softmax/softmax_columns` `test_case[sweep_08[rows=10,width=129,scale=0.01,seed=7]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=10,width=129,scale=0.01,seed=7` |
| `test_case[softmax_columns-rows=256,width=256,scale=3.0,seed=8]` | `//kernels/softmax/softmax_columns` `test_case[sweep_09[rows=256,width=256,scale=3.0,seed=8]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=256,width=256,scale=3.0,seed=8` |
| `test_case[softmax_columns-rows=32,width=300,scale=4.0,seed=9]` | `//kernels/softmax/softmax_columns` `test_case[sweep_10[rows=32,width=300,scale=4.0,seed=9]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=32,width=300,scale=4.0,seed=9` |
| `test_case[softmax_columns-rows=16,width=460,scale=3.0,seed=10]` | `//kernels/softmax/softmax_columns` `test_case[sweep_11[rows=16,width=460,scale=3.0,seed=10]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=16,width=460,scale=3.0,seed=10` |
| `test_case[softmax_columns-rows=8,width=384,scale=50.0,seed=11]` | `//kernels/softmax/softmax_columns` `test_case[sweep_12[rows=8,width=384,scale=50.0,seed=11]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=8,width=384,scale=50.0,seed=11` |
| `test_case[softmax_columns-rows=32,width=65,scale=4.0,seed=12]` | `//kernels/softmax/softmax_columns` `test_case[sweep_13[rows=32,width=65,scale=4.0,seed=12]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=32,width=65,scale=4.0,seed=12` |
| `test_case[softmax_columns-rows=32,width=96,scale=3.0,seed=13]` | `//kernels/softmax/softmax_columns` `test_case[sweep_14[rows=32,width=96,scale=3.0,seed=13]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=32,width=96,scale=3.0,seed=13` |
| `test_case[softmax_columns-rows=32,width=127,scale=5.0,seed=14]` | `//kernels/softmax/softmax_columns` `test_case[sweep_15[rows=32,width=127,scale=5.0,seed=14]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=32,width=127,scale=5.0,seed=14` |
| `test_case[softmax_columns-rows=16,width=100,scale=50.0,seed=15]` | `//kernels/softmax/softmax_columns` `test_case[sweep_16[rows=16,width=100,scale=50.0,seed=15]-fp32]`; `compare_legacy` row `softmax_columns` / `rows=16,width=100,scale=50.0,seed=15` |

### Tests moved out of `src/tools/ipu-apps/test/test_kernel_registry.py` (E07-11)

The file stays. Its registry-mechanics tests that used softmax as the example now use `identity` (the one old kernel E07-04 keeps until E07-25) where a real kernel is needed (discovery, a missing parameter, derived shapes, layer lookup, the benchmark runner, the query CLI, the constructor guard), and synthetic specs (the file's `_registered`) where several kernels are needed: `SPLIT`, three kernels split by row length as the softmax row kernels were (overlap resolved by cost, routing, boundaries, aggregated and deduplicated refusals), and a flattening spec for the rank-3 disclosure. `test_discovery_finds_the_softmax_kernels` became `test_discovery_finds_the_identity_kernel`; the other names stay. These left it:

| Test | Now covered by |
|---|---|
| `test_softmax_neighbours_are_refused` (2 ids) | `kernels/softmax/test_layers.py::test_softmax_relatives_are_refused_by_name` (the new adapters, `ipu_kernels/softmax/layers.py`) |
| `test_softmax_without_an_explicit_dim_is_refused` | `kernels/softmax/test_layers.py::test_softmax_without_a_dim_is_refused` |
| `test_resolved_kernel_computes_the_operation` (28 ids) | `kernels/softmax/test_conformance.py::test_the_resolved_kernel_computes_softmax` (the same 14 shapes x 2 dims and data, through `Registry.resolve` and `run_kernel`) |
| `test_query_rejects_malformed_queries[argv1]` (`dim=5`: the resolver's `ValueError`) | the same test's `r in two parameters` id: an ambiguous `--sweep` raises the `ValueError` the same `except` turns into exit 2 (no `identity` query raises one) |

### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/<fold or unfold kernel>/test.py` (6 files)

| Test | Now covered by |
|---|---|
| `test_case` (`case_tests(__package__)`: every case in the folder's `cases.py`) | The kernel's generated tests, `bazel test //kernels/reshape/<kernel>`: the `default` case (and unfold_8x8x240's `garbage_padding`) with the old data (`../gen.py`), held EXACT to `ops.unfold`/`ops.fold`. |
| `unfold_8x8x240`: `test_output_shape_and_stale_lanes` | `kernels/reshape/test_fold_unfold.py::test_unfold_8x8x240_stores_one_channel_per_row_and_a_zero_tail` |
| `unfold_8x8x240`: `test_padding_is_inert` | The `garbage_padding` case of `kernels/reshape/unfold_8x8x240/kernel.toml`: its output must equal the reference of the same tensor, as the zero-padded case's does. |

### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/reshape/test.py`

| Test | Now covered by |
|---|---|
| `test_fold_inverts_unfold` (3 layers x plain, poisoned R_ACC) | `kernels/reshape/test_fold_unfold.py::test_fold_inverts_unfold`, same seeds, unfold's output region fed to fold on one model; R_ACC poisoned by a host write between the steps. |
| `test_fold_and_unfold_share_their_shapes` | `kernels/reshape/test_fold_unfold.py::test_fold_and_unfold_share_their_shapes`, and `kernels/reshape/test_routing.py::test_each_fold_and_unfold_is_routed_by_its_exact_shape` |
### Tests from `src/tools/ipu-apps/src/ipu_apps/kernels/conformance/vector_tour/test.py`

| Test | Now covered by |
|---|---|
| `test_case` (`case_tests(__package__)`: every case in `cases.py`: `default`, `int8`, `e4`, `e5`, `e3`) | `bazel test //kernels/conformance/vector_tour` (`test_case[<case>-<format>]`, the same five cases on the old data, `gen.py:legacy_steps`), against `reference.py` at `EXACT`. |
