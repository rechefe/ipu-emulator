# Create the `ipu_debug` package and move the debugger into it

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-03 · Size M · Depends on: E08-02, E03 (package-split pattern and import-graph test, Q62) · Behaviour change: none

## Goal

The debugger becomes its own package, `src/tools/ipu-debug/src/ipu_debug/` (Q69), and depends one way on `ipu_emu` and `ipu_isa`. This is a mechanical move with no logic change. Later issues refactor it inside the new home.

## Background

- The debugger modules live in `ipu_emu`:
  - `debug_cli.py` (1165 lines), `debug_tui.py` (3732), `debug_control.py`, `debug_launch.py` and `debug_pipeline.py`;
  - tests `test_debug_cli.py` and `test_debug_tui.py` in `ipu-emu-py/test/`. `test_wide_vector_debug.py` is about the emulator's wide-vector mode, not the debugger, and stays.
- External callers:
  - `ipu_apps/kernel_registry/base.py:93` lazily imports `ipu_emu.debug_launch`;
  - `ipu-apps/BUILD.bazel:100` (`test_debug_launch`);
  - `.bazelrc:28-32` (`run:debug` sets `IPU_DEBUG_TUI=1`).
- Q62: one-way deps, enforced by Bazel deps and the import-graph test, with no cycle-dodging lazy imports. The lazy import at `base.py:93` stays allowed only because it is a real optional dependency (debug mode), not a cycle. E06's "runner owns debug" decides its final shape.

## Implementation

1. Create `src/tools/ipu-debug/` with `BUILD.bazel` and `pyproject.toml`, following the layout E03 sets for `ipu_isa`. Target `//src/tools/ipu-debug:ipu_debug_lib` depends on `ipu_emu_lib` and `ipu_isa_lib`.
2. `git mv` the modules, renaming them inside the package:
   - `debug_cli.py` → `ipu_debug/cli.py`
   - `debug_tui.py` → `ipu_debug/tui.py` (split per pane later)
   - `debug_launch.py` → `ipu_debug/launch.py`
   - `debug_pipeline.py` → `ipu_debug/pipeline.py`
   - the frontend half of `debug_control.py` → `ipu_debug/control.py`. The run-loop half stays in `ipu_emu` (E08-02).
3. Move the tests to `src/tools/ipu-debug/test/` with their Bazel targets, keeping the target names (`test_debug_cli`, `test_debug_tui`).
4. Update `kernel_registry/base.py` to import `ipu_debug.launch`, and add `ipu_debug_lib` to the ipu-apps deps.
5. Register the package in `tools/import_graph.toml` and add its `:py_sources` to `//tools:test_import_graph`: `ipu_debug` may import `ipu_emu` and `ipu_isa`, and nothing may import `ipu_debug` except `ipu_apps` / `ipu_harness` (the runner).
6. Update `docs/content/debugging.md` paths only. No content rewrite (E09).

## Tests

- [ ] All moved tests pass unchanged apart from import paths. The test count is identical; the PR lists before and after counts.
- [ ] Import-graph test: a planted `import ipu_debug` inside `ipu_emu` fails it.
- [ ] Manual smoke: `bazel run --config=debug //src/tools/ipu-apps:identity` opens the TUI, steps, quits. Noted in the PR.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] No `debug_*.py` remains in `ipu_emu`.
- [ ] `ipu_emu_lib` has no Bazel dep on `ipu_debug_lib`.
