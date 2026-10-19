# `diagnostics.check` runs on the session and reads error locations

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-07 · Size M · Depends on: E05-05, E05-08 (contract snapshot) · Behaviour change: two `check --json` columns move to the offending token (see the E05-07 commits)

## Goal

`diagnostics.check` (the engine behind `ipu-as check --json` and the VS Code squiggles) stops re-implementing the assembly pipeline. It drives the stages of `AssemblySession` and reads `error.location` / `error.token` from the typed errors (E05-04) instead of regex-parsing message text. Only `AssemblerError` becomes a user diagnostic. Any other exception is a bug and goes to the existing "internal" path.

## Background

- `diagnostics.check` (`ipu_as/diagnostics.py:468-532`) has its own render → parse (`get_parser().parse`) → `ASTBuilder().transform` → `CompoundInst(...).encode()` loop, and resets the global label table (`:508`). `_recover_line_without_fusion` (`:196-245`) runs a second copy (reset at `:215`).
- Positions come from `_LOCATION_RE` (`:36`) and `_QUOTED_NAME_RE` (`:41`), applied to the message (`:233, 403-404`).
- `except ValueError` (`:231, 529`) turns any `ValueError`, including a programming bug inside the encoder, into a user diagnostic.
- The instrumented render with `#@ipu-src-line:` markers (`:74-130`), which maps rendered positions back to the source, is valuable and stays.
- `cli.check` (`cli.py:62-122`) already maps unexpected exceptions to an `internal` diagnostic (`stage="internal"`).

## Implementation

1. Add stage-level entry points to `AssemblySession` (or a `DiagnosticRun` adapter over it) that accept an already-rendered text, so diagnostics can pass the instrumented render: `session.parse_rendered(text)` and `session.encode(parsed)`.
2. Rewrite `check(text, *, defines=None)`:
   - keep the template stage, including the instrumented render and its fallback;
   - catch `TemplateError`, `ParseError` and `EncodeError` (with `LabelError`, `OperandError` and `SlotError`), not `ValueError` or `Exception`;
   - build the `Diagnostic` from `error.location` (1-based; convert to 0-based) and `error.token`, then map rendered to source positions with the existing marker logic.
3. Delete `_LOCATION_RE` and `_QUOTED_NAME_RE`. Any error site that still lacks a location is a gap to fix in E05-04 terms (give it a location), not a regex to keep.
4. `_recover_line_without_fusion` uses a fresh session instead of `reset_labels`.
5. `check` accepts `defines`, forwarded to the session. The CLI flag is wired in E05-10.

## Tests

- [ ] Every existing `test_diagnostics` case passes unchanged.
- [ ] The `check --json` v1 snapshot from E05-08 is byte-identical on the extension corpus (`vscode-ipu-asm/test/corpus`).
- [ ] A monkeypatched encoder that raises a plain `ValueError` yields an `internal` diagnostic, not an `assemble` one.
- [ ] Two consecutive `check` calls with the same label names give no "defined for the second time" error, with no reset call.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `diagnostics.py` contains no parse/transform/encode loop of its own and no regex over error messages.
- [ ] Only `AssemblerError` subclasses become user diagnostics.
