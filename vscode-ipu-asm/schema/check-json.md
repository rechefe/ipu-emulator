# The `ipu-as check --json` contract

The extension runs the checkout's own assembler, `ipu-as check --json --input -`,
and reads what it prints. The extension is published separately from the
repository, so an extension of one age meets assemblers of every age. This file
says what the output is and how it may change.

## Versions

| Version | Asked for with | Output | Schema |
|---|---|---|---|
| 1 | `--json` (or `--json --json-version 1`) | a bare array of diagnostics | [check-json-v1.schema.json](check-json-v1.schema.json) |
| 2 | `--json --json-version 2` | `{"schemaVersion": 2, "diagnostics": [...]}`, the same diagnostics, and the analyses asked for (below) | [check-json-v2.schema.json](check-json-v2.schema.json) |

`ipu-as check --json-versions` prints the versions the assembler supports, as a
JSON array (`[1, 2]`). An assembler from before versions rejects that option, and
`--json-version`, with a usage error (exit code 2): it supports version 1 only.

A diagnostic has `line`, `column`, `end_line` and `end_column` (all from 0),
`message`, `severity` (`error`), `stage` (`template`, `parse`, `assemble`, or
`internal` when the checker itself failed) and `approximate` (the position could
not be pinned down exactly). The exit code is 1 when there are diagnostics and 0
when there are none.

## Analyses (version 2)

With `--json --json-version 2`, three options each add one more analysis of the
same source, under the key of the same name; `diagnostics` is always there, and a
key whose option is not given is absent. One call answers every question the
editor asks of a document version:

    ipu-as check --json --json-version 2 --tokens --words --jinja --input -

| Option | Key | Each entry |
|---|---|---|
| `--tokens` | `tokens` | A semantic token: `line`, `col`, `length`, `type` and `role` (`mnemonic`, `register`, `enum`, `number`, `label-def`, `label-ref`, `comment` or `jinja`). |
| `--words` | `words` | A word's slot fit: `range`, `mnemonics`, `slots` (slot name to count), `fits` and `error` (the assembler's message when it does not fit, else `null`). |
| `--jinja` | `jinja` | A Jinja tag: `kind` (`comment`, `block` or `variable`), `range`, the names it `defines` and `uses`, and its `start` and `end` offsets. |

The options with version 1, or with `--text`, are a usage error (exit code 2).
`ipu_as.language_service` computes them, from the assembler's own lexer, slot
assignment and render, and from Jinja's own lexer and parser:

- Tokens are in source positions, on one line each, sorted and disjoint. An
  assembly token that a `{{ }}` expression produced is not in the source, so it
  is not listed; the `jinja` token over the expression is. A token a loop
  renders several times is listed once. Separators (`;`, `;;`, `:`) and
  whitespace are not listed.
- Each word is fitted on its own, so one that does not fit hides no other. Its
  `range` runs from its first token in the source to the end of its last. A
  word that renders the same way in the same place is listed once.
  `mnemonics` are as written; `slots` counts after pseudo-instructions expand.
  There are no words when the source does not render or parse.
- A Jinja region's `defines` are the names a `set`, `for`, `macro` (and its
  parameters) or `import` binds, and for the `ipu:params` header comment the
  parameters it declares. Its `uses` are the names it reads, Jinja's globals
  (`range`, `loop`) included. An unclosed tag runs to the end of the source.
  `{% raw %}` and `{% endraw %}` are block regions; the text between them is
  not a region.

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
