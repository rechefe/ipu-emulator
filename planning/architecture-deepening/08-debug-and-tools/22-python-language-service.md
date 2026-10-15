# Python language service behind `check --json` v2

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-22 · Size M · Depends on: E05-07 (diagnostics on the session), E05-08 (versioned `check --json`) · Behaviour change: none (v2 adds fields; v1 unchanged)

## Goal

The checkout's assembler answers every analysis question the editor asks, in one process per document version:

- diagnostics;
- semantic tokens (type and role);
- per-word slot fit;
- Jinja regions with the names they define and use.

`ipu_as/language_service.py` is a deep module over `AssemblySession`. `ipu-as check --json-version 2` is its adapter. There is no LSP (L37).

## Background

- `vscode-ipu-asm/language.js` re-implements, by hand:
  - the scanner (mirroring `asm_grammar.lark`);
  - slot fit (`fits`, `language.js:345-359`);
  - Jinja scoping (`jinjaRegions` `:19`, `jinjaDefs` `:457`, `jinjaNames` `:511`).
- The agreement tests check only one direction, and the Jinja "oracle" in `ipu_as/dump_parser_tokens.py:55-76` copies the JS algorithm, so it cannot catch JS bugs.
- E05-08 defines `--json-version 2` as `{"schemaVersion": 2, "diagnostics": [...]}` and allows only additive fields.

## Implementation

```python
# ipu_as/language_service.py
@dataclass(frozen=True)
class Token: line: int; col: int; length: int; type: str; role: str       # role: mnemonic|register|label-def|label-ref|number|jinja|comment
@dataclass(frozen=True)
class WordFit: range: Range; mnemonics: tuple[str, ...]; slots: Mapping[str, int]; fits: bool; error: str | None
@dataclass(frozen=True)
class JinjaRegion: kind: str; range: Range; defines: tuple[str, ...]; uses: tuple[str, ...]

def analyse(text: str, *, defines=None, search_path=(), want=frozenset({"diagnostics"})) -> Analysis
```

1. **Tokens** come from the lark lexer on the rendered text, mapped back to the source with the E05-07 line markers. Template regions come from Jinja's own lexer (`Environment.lex`), not a copy of the JS algorithm.
2. **Words** run the slot assignment from E05-02 on each parsed word independently, so one bad word does not hide the others.
3. **Jinja**: `defines` from `{% set %}`, `{% for %}`, `{% macro %}` and `ipu:params`; `uses` from Jinja's AST (`jinja2.meta`-style traversal).
4. CLI: `check --json-version 2 [--tokens] [--words] [--jinja] --input -`. Each flag adds its key; `diagnostics` is always present. Update the v2 JSON schema file (E05-08) with the new optional keys.
5. Replace the Jinja section of `dump_parser_tokens.py` with a call to `language_service`, so the agreement test checks JS against the real implementation.

## Tests

- [ ] Python unit tests per feature on the extension corpus (`test/corpus/edge_cases.asm`) and on the kernel sources.
- [ ] v2 output validates against the schema. `diagnostics` equals v1 for the same input.
- [ ] Performance: `analyse` of the largest kernel source with all features takes under 300 ms on CI (tracked number).
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] One process call answers every analysis question for a document version.
- [ ] The agreement oracle no longer contains a copy of the JS Jinja algorithm.
