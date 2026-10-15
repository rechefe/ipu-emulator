# Pin and version the `check --json` contract

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-08 · Size S · Depends on: E02 · Behaviour change: none (v1 output stays byte-identical)

## Goal

The VS Code extension (published on the Marketplace and also run from a checkout) parses `ipu-as check --json`. That makes the output an external contract. Pin it with a snapshot test and give it an integer schema version (L40), so that later work (E05-07 in this epic; language services in E08) can add to it without breaking an older extension.

## Background

- `cli.check` (`ipu_as/cli.py:62-122`) writes a bare JSON array of `Diagnostic.to_dict()` (`diagnostics.py:62-63`). The keys are `line`, `column`, `end_line`, `end_column`, `message`, `severity`, `stage` and `approximate`.
- Consumers:
  - `vscode-ipu-asm/checker.js:130` spawns `check --json --input -`;
  - `extension.js:283-294` reads the position keys, `approximate` and `stage`;
  - `editing.js:179` builds a key from `stage` plus `message`, so message wording is part of the contract.
- There is no version marker, so a newer checkout cannot tell an older extension what it emits.

## Implementation

1. Add `vscode-ipu-asm/contract/check-json-v1.schema.json`, a JSON Schema for today's array. Add `check-json.md` next to it describing the contract rules:
   - v1 = today's bare array;
   - a new version may only add fields;
   - a message-text change is a contract change;
   - the extension supports the current and the previous version.
2. Add a CLI option `--json-version N` (default `1`):
   - `1` prints exactly today's array;
   - `2` prints `{"schemaVersion": 2, "diagnostics": [...]}` with the same diagnostic objects. E08-22 adds `tokens`, `words` and `jinja` under v2.

   An old `ipu-as` rejects the unknown option with a usage error (exit code 2). The extension treats that as "v1 only" and falls back (E08-23).
3. Add `ipu-as check --json-versions`, which prints `[1, 2]`, so the extension can ask without guessing.
4. Snapshot test: run `check --json` over `vscode-ipu-asm/test/corpus/**/*.asm` plus the `test_diagnostics` sources, and compare against a checked-in `test/golden/check_json_v1.json`. Validate the output against the schema file.

## Tests

- [ ] The v1 snapshot equals the base-commit output (recorded first, on the base commit).
- [ ] The v2 output validates against `check-json-v2.schema.json`, and its `diagnostics` equal the v1 array.
- [ ] `--json-version 3` exits 2 with a usage message.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] The contract files exist and are referenced from the extension README (link only; E08 rewrites the README).
- [ ] Any change to v1 output fails the snapshot test.

## Open question

- Do the schema files belong under `vscode-ipu-asm/contract/` or under `ipu_as`? The assumption here is the extension folder, because the extension owns the consumer side. The Python test reads them as Bazel `data`.
