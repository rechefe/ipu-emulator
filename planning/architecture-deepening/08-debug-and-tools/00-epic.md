# Epic: Debugger, generators and editor as deep modules with thin adapters

ID E08 · Folder `08-debug-and-tools` · Runs alongside E04-E07 once its prerequisites land (see the per-issue dependencies).

## Goal

Four tool areas get the same shape: **one deep module, thin adapters**.

1. **`ipu_debug`** (Q69, Q67). A `DebugSession` core has no curses and no I/O. It owns parsing, register and XMEM views, disassembly, the pipeline view and every mutation (register set, breakpoints, run-until). The line CLI and the curses TUI are adapters over it. The TUI is split into one module per pane.
2. **`ipu_gen`** (Q69). The generators for docs pages, the SystemVerilog package, VS Code language data and union-layout SVGs move out of `ipu_as` and `ipu_common` into one package. The `*_wrapper.py` shims go.
3. **Cycle Inspector in the docs** (Q55, phase 1). The trace viewer becomes a docs page fed by traces that Bazel generates with the E04 reference-model trace writer. They are regenerated on every docs build. Live mode (phase 2) is a follow-up only.
4. **VS Code extension.**
   - Security and state fixes land first, as small issues.
   - Then the assembler in the user's checkout becomes the single source of language data and analysis, behind a versioned `check --json` contract.
   - The extension stays its own JavaScript project, outside Bazel (D9).

## Background

Verified at HEAD `694a8a8`.

- **Debugger.** The debugger lives inside `ipu_emu`.
  - `debug_tui.py` is 3732 lines. `CursesDebugView` (`debug_tui.py:779-3703`) has 130 methods, and `handle_key` (`:2539-2683`) and `_draw_xmem` (`:1775-1915`) are the longest functions.
  - `debug_tui.py:27-40` imports 11 private `_CELL16_*` and width constants from `debug_cli`, and lazy imports pull in `_parse_int`, `_resolve_xmem_request`, `_tokenize_xmem` and `_parse_xmem_request` (`:568, 1334, 1344, 3099, 3159, 3201, 3322`).
  - Register set exists twice and has diverged (`debug_cli.py:948` `do_set`, `debug_tui.py:3200` `_submit_scalar_editor`).
  - `do_delete` mutates `control.breakpoints` directly (`debug_cli.py:867, 872`), and `set pc` writes the PC unchecked (`debug_cli.py:960-961`).
  - `DebugControl` is attached to the state through a module-global `WeakKeyDictionary` (`debug_control.py:46-54`), and `emulator.run_with_debug` reads it (`emulator.py:23, 115`).
  - `debug_launch.py:26` raises `SystemExit(0)` inside the emulator callback.
  - `ipu_emu/__init__.py:25-32` re-exports `debug_cli` names, so the emulator package imports the debugger.
  - `test_debug_tui.py` has 3139 lines and 153 `monkeypatch` uses.
- **Generators.**
  - `gen_docs.py`, `gen_codegen.py` (plus `templates/ipu_instr_pkg.sv.j2`) and `gen_vscode.py` live in `ipu_as`.
  - `union_layout_svg.py` and `gen_union_layout_svg.py` live in `ipu_common`.
  - `compound_inst.py:179-260` renders layout SVGs.
  - Seven `*_wrapper.py` scripts wrap them (six are thin re-exports; `serve_docs_wrapper.py` is a small real script): `docs/gen_docs_wrapper.py`, `docs/gen_codegen_wrapper.py`, `docs/gen_instruction_format_md_wrapper.py`, `docs/mkdocs_wrapper.py`, `docs/serve_docs_wrapper.py`, `vscode-ipu-asm/gen_vscode_wrapper.py` and `vscode-ipu-asm/dump_parser_tokens_wrapper.py`.
  - `gen_vscode_wrapper.py:23` reaches into the private `Ipu._lrd_lr_indices`.
- **Cycle Inspector.** The artifact (https://claude.ai/artifact/4sTeRLzw27gc3z8JmhhMVB) is a static page. It fetches `data/isa.json` and `data/<kernel>.json` traces, each with:
  - `meta`, with dtype and wide mode;
  - `program`: per word, the label, the hex word and the decoded slots;
  - `layout`: field names and widths;
  - `init`: hex register images;
  - `cycles`: per cycle, the PC, register deltas and lanes used;
  - `total_cycles`, `truncated` and `source`.

  The 14 traces were recorded ad hoc. Nothing in the repo produces them.
- **VS Code.**
  - `package.json` has no `untrustedWorkspaces` declaration. `ipuAsm.checkCommand`, `manifestCommand` and `queryCommand` can be set per workspace and run automatically.
  - `runs.js:67-68, 81` pass manifest fields (`targets.*`, `family.suite`) unquoted into `terminal.sendText` (`kernels.js:150`).
  - Module-level state sits at `extension.js:30-46`.
  - `checked.set` runs before the error branch (`extension.js:308` vs `:318`).
  - Non-file URIs go to the first checkout (`extension.js:209-213`).
  - Hover data comes from the bundled `isa.json` (`extension.js:83-85`), while diagnostics come from the checkout's `ipu-as check --json` (`checker.js:130`), so the two can disagree on a branch.
  - `language.js` re-implements the scanner, slot fit, number syntax, Jinja scoping and the formatter.

## Decisions applied

- **Q69**: `ipu_debug` package (`DebugSession` core plus CLI and TUI adapters). `ipu_gen` package (`gen_docs`, `gen_codegen`, `gen_vscode` and the union-layout SVG move out of `ipu_as` and `ipu_common`). The `*_wrapper.py` shims are removed. Each package is one directory under `src/tools/`.
- **Q62**: One responsibility per package, one-way dependencies, enforced by Bazel deps and the import-graph test, with no cycle-dodging lazy imports.
- **Q67**: TUI split per pane over `DebugSession`.
- **L35**: Keep both CLI and TUI, as adapters.
- **L36**: The formatter moves to Python as `ipu-as format`.
- **L37**: No LSP for now.
- **L38**: Support both Marketplace and checkout users; checkout data is preferred, bundled data is the fallback.
- **L39**: Restricted Mode is `limited`.
- **L40**: Integer `schemaVersion`.
- **D9**: JavaScript stays out of Bazel, and the extension is its own project. The vscode workflow's path triggers are handled in E01.
- **Q55**: The Cycle Inspector becomes a docs page fed by Bazel-generated traces from the E04 trace writer. Phase 2 (Pyodide live mode) is a follow-up issue only, outside the program's scope.
- **D3**: Refactors are output-identical, and behaviour changes are labelled.
- **D5**: Only the reference-model API is stable. Debugger internals may change freely.

## Scope

- `src/tools/ipu-emu-py/src/ipu_emu/debug_*.py` → `src/tools/ipu-debug/src/ipu_debug/`
- Generators → `src/tools/ipu-gen/src/ipu_gen/`
- `docs/` build rules for the generated pages and the Cycle Inspector page
- `vscode-ipu-asm/**` (JavaScript and its Python helpers)

## Out of scope

- The decoded-word codec (E03); this epic consumes it.
- The reference-model trace format and writer (E04); this epic consumes it.
- The runner's debug and profile flags (E06 "runner owns debug/profile"); E08-06 provides the hook it calls.
- Docs restructure and page rewrites (E09). E08 updates only the pages its own changes break (`debugging.md` commands, the extension README).
- Generated-docs conventions fixes (upper-case registers, element units in `gen_docs` output): E09, after the E08-14 move.
- Cycle Inspector phase 2 (live Pyodide mode): a follow-up issue (E08-18, filed as #256), not in this program.
- LSP (L37).

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|----|------|-------|------|-----------|-------------------|
| E08-01 | [01-breakpoint-validation.md](01-breakpoint-validation.md) | Validate every breakpoint and PC mutation in `DebugControl` | S | — | yes (CLI `set pc` with an out-of-range value reports an error) |
| E08-02 | [02-emulator-debug-seam.md](02-emulator-debug-seam.md) | Explicit debug seam in the emulator: no `WeakKeyDictionary`, no debugger re-exports | S | E08-01 | none |
| E08-03 | [03-ipu-debug-package.md](03-ipu-debug-package.md) | Create the `ipu_debug` package and move the debugger into it | M | E08-02, E03 (package split pattern, import-graph test) | none |
| E08-04 | [04-debug-session-core.md](04-debug-session-core.md) | `DebugSession` core: public parse, resolve and read API | M | E08-03 | none |
| E08-05 | [05-one-mutation-path.md](05-one-mutation-path.md) | One mutation path: register set, breakpoints and run-until in `DebugSession` | M | E08-04 | yes (CLI `set` clears XMEM caches; TUI editor accepts `pc`) |
| E08-06 | [06-quit-without-exit.md](06-quit-without-exit.md) | QUIT without `SystemExit`; kernel name passed explicitly | S | E08-04 | none |
| E08-07 | [07-pipeline-from-decoded-word.md](07-pipeline-from-decoded-word.md) | Pipeline view from the structured decoded word | M | E08-04, E03 (codec `DecodedWord`) | none |
| E08-08 | [08-tui-shell.md](08-tui-shell.md) | TUI shell: `TabSet`, screen adapter and keymap | M | E08-05 | none |
| E08-09 | [09-tui-xmem-pane.md](09-tui-xmem-pane.md) | TUI XMEM pane module | M | E08-08 | none |
| E08-10 | [10-tui-registers-pane.md](10-tui-registers-pane.md) | TUI registers pane module | M | E08-08 | none |
| E08-11 | [11-tui-disassembly-pipeline-panes.md](11-tui-disassembly-pipeline-panes.md) | TUI disassembly and pipeline pane modules | M | E08-08, E08-07 | none |
| E08-12 | [12-tui-error-hygiene.md](12-tui-error-hygiene.md) | TUI error hygiene: no `except Exception: pass` | S | E08-09, E08-10, E08-11 | yes (unexpected errors are logged, not swallowed) |
| E08-13 | [13-ipu-gen-codegen.md](13-ipu-gen-codegen.md) | Create `ipu_gen`; move the SystemVerilog generator | M | E03 (package split), E02-03 (SV golden) | none |
| E08-14 | [14-ipu-gen-docs-svg.md](14-ipu-gen-docs-svg.md) | Move the docs-page and union-layout SVG generators into `ipu_gen` | M | E08-13 | none |
| E08-15 | [15-ipu-gen-vscode-no-wrappers.md](15-ipu-gen-vscode-no-wrappers.md) | Move `gen_vscode` into `ipu_gen`; delete every `*_wrapper.py` shim | S | E08-14 | none |
| E08-16 | [16-inspector-trace-targets.md](16-inspector-trace-targets.md) | Bazel targets that record Cycle Inspector traces | M | E04-29 (trace schema), E04-30, E08-15 | none |
| E08-17 | [17-inspector-docs-page.md](17-inspector-docs-page.md) | Cycle Inspector docs page (phase 1) | M | E08-16 | none (new page) |
| E08-18 | [18-inspector-live-mode-followup.md](18-inspector-live-mode-followup.md) | Follow-up: Cycle Inspector live mode (Pyodide), out of program scope | — | E08-17 | n/a |
| E08-19 | [19-vscode-workspace-trust.md](19-vscode-workspace-trust.md) | VS Code: Restricted Mode support and quoted manifest fields | S | — | yes (no configured command runs in an untrusted workspace) |
| E08-20 | [20-vscode-state-bugs.md](20-vscode-state-bugs.md) | VS Code: per-checkout state and retry after a failed check | S | — | yes (bug fixes) |
| E08-21 | [21-number-and-slot-rules.md](21-number-and-slot-rules.md) | One source for number syntax and slot-fill rules | M | E08-15 | yes (editor accepts signed hex exactly when the assembler does) |
| E08-22 | [22-python-language-service.md](22-python-language-service.md) | Python language service behind `check --json` v2 | M | E05-07, E05-08 | none (v2 is additive) |
| E08-23 | [23-extension-analysis-client.md](23-extension-analysis-client.md) | Extension consumes `check --json` v2; delete JS slot fit and Jinja scoping | M | E08-22, E08-21 | yes (analysis comes from the checkout's assembler) |
| E08-24 | [24-ipu-as-format.md](24-ipu-as-format.md) | `ipu-as format` and the extension's format provider | M | E08-22 | yes (formatting done by Python) |
| E08-25 | [25-language-data-from-checkout.md](25-language-data-from-checkout.md) | Language data from the checkout, bundled data as fallback | M | E08-15, E08-23 | yes (hover and completion follow the checkout) |
| E08-26 | [26-extension-structure.md](26-extension-structure.md) | Extension structural cleanup | M | E08-20 | none |
| E08-27 | [27-extension-tests.md](27-extension-tests.md) | Extension tests: negative agreement corpus, no fixed sleeps | S | E08-21 | none |

## Suggested order

```
Debugger:   01 ──▶ 02 ──▶ 03 ──▶ 04 ──▶ 05 ──▶ 08 ──▶ 09 ─┐
                                  ├──▶ 06        ├──▶ 10 ──┼──▶ 12
                       E03 ──▶    └──▶ 07 ───────┴──▶ 11 ──┘
Generators: E03,E02 ──▶ 13 ──▶ 14 ──▶ 15
Inspector:  E04-29/30 + 15 ──▶ 16 ──▶ 17 ┄┄▶ 18 (follow-up, out of scope)
VS Code:    19, 20 first (any time)
            15 ──▶ 21 ──▶ 27
            E05-07,E05-08 ──▶ 22 ──▶ 23 ──▶ 25
                               └──▶ 24
            20 ──▶ 26
```

E08-01, E08-19 and E08-20 are small and independent. File and land them first.

## Epic acceptance

- [ ] `ipu_emu` imports nothing from `ipu_debug`, and the import-graph test enforces it. Library code never raises `SystemExit`.
- [ ] Register set, breakpoint add/remove/clear and run-until each exist once and validate the PC.
- [ ] No TUI function is longer than about 80 lines. No `except Exception: pass` remains. `test_debug_tui.py` has fewer than 30 `monkeypatch` uses and no private-name calls.
- [ ] `ipu_as` and `ipu_isa` contain no generator. No `*_wrapper.py` remains. The SV package, `isa.json` and generated pages are byte-identical to the E02 goldens.
- [ ] The docs site has a Cycle Inspector page whose traces are rebuilt by Bazel on every docs build.
- [ ] With a checkout present, hover, completion, slot fit, formatting and diagnostics all come from the checkout's assembler. Restricted Mode runs no configured command.
