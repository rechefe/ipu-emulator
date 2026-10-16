# The `ipu-as check --json` contract

The extension runs the checkout's own assembler, `ipu-as check --json --input -`,
and reads what it prints. The extension is published separately from the
repository, so an extension of one age meets assemblers of every age. This file
says what the output is and how it may change.

## Versions

| Version | Asked for with | Output | Schema |
|---|---|---|---|
| 1 | `--json` (or `--json --json-version 1`) | a bare array of diagnostics | [check-json-v1.schema.json](check-json-v1.schema.json) |
| 2 | `--json --json-version 2` | `{"schemaVersion": 2, "diagnostics": [...]}`, the same diagnostics | [check-json-v2.schema.json](check-json-v2.schema.json) |

`ipu-as check --json-versions` prints the versions the assembler supports, as a
JSON array (`[1, 2]`). An assembler from before versions rejects that option, and
`--json-version`, with a usage error (exit code 2): it supports version 1 only.

A diagnostic has `line`, `column`, `end_line` and `end_column` (all from 0),
`message`, `severity` (`error`), `stage` (`template`, `parse`, `assemble`, or
`internal` when the checker itself failed) and `approximate` (the position could
not be pinned down exactly). The exit code is 1 when there are diagnostics and 0
when there are none.

## Rules

- Version 1 is the bare array, and it never changes: a snapshot test
  (`//src/tools/ipu-as-py:test_check_json`) fails on any difference.
- A new version may only add fields. Nothing a version prints is removed or
  changes meaning in a later one. Each version has a schema of its own: a
  version that adds a field to each diagnostic defines its own diagnostic,
  since version 1's allows no other fields.
- A diagnostic's message text is part of the contract: the extension compares
  messages (a rename is refused only for a problem the renamed text adds). A
  change to a message is a contract change.
- The extension supports the current version and the one before it.
