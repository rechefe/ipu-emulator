# GNU numeric local labels (`1:`, `1f`, `1b`)

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-14 · Size S · Depends on: E05-05 · Behaviour change: yes (new syntax; every existing program assembles unchanged)

## Goal

Let a macro or loop body use short labels that may be defined many times, without uniquifying names by hand (L7). The syntax is the GNU assembler one: `N:` defines a local label N (decimal digits only). `Nf` refers to the next `N:` after the current word, and `Nb` refers to the nearest `N:` at or before the current word.

## Background

- Labels live in one flat table, which E05-05 replaces with `SymbolTable`. A second definition is an error (`label.py:13-21`).
- Macros uniquify labels by pasting a template variable into the name, for example `ch_loop_stripe{{ stripe }}:` and its branch at `kernels/reshape/fold_32x32x144/fold_32x32x144.asm:224, 266`.
- No grammar change is needed: `TOKEN` (`asm_grammar.lark:23`, `/[+-]?[A-Za-z0-9_][A-Za-z0-9_\.]*/`) already lexes `1`, `1f` and `1b`, and `label: TOKEN ":"` accepts `1:`.
- No source in the tree defines a digit-only label today (checked with `grep -rnE "^\s*[0-9]+:"`), so making digit-only labels local changes no existing program.
- `LabelToken` (`ipu_token.py:155-199`) handles `+N` relative targets and named labels.

## Implementation

1. `SymbolTable` keeps local labels as `dict[int, list[int]]` (label number → sorted addresses). `define` allows repeats for digit-only names and still rejects repeats for named labels.
2. Resolution in `SymbolTable.address(ref, at=word_index)`:
   - `Nf`: the smallest address strictly greater than `at`.
   - `Nb`: the largest address less than or equal to `at`. A word may branch to its own label with `1b`, as in GAS, where a label marks the current location.
   - No match: `LabelError("local label 1f has no definition after line L")`, with the location.
3. A bare `1` as a branch target stays an error with a hint: "use 1f or 1b". A bare digit is ambiguous with immediates in other operand positions.
4. Editor support, in the same PR so the editor does not flag valid code:
   - `gen_vscode._label_pattern` (`gen_vscode.py:205`) is derived from the grammar, so the TextMate label rule needs no change. Add `Nf` / `Nb` reference highlighting if the reference pattern is listed separately.
   - Check that `language.js` rename and "go to definition" skip local labels: renaming `1` is meaningless. Coordinate with E08-23 if JS changes are needed.
5. Convert `fold_32x32x144.asm` as the demonstration in a follow-up commit of the same PR. Its digest must be unchanged.

## Tests

- [ ] Two macro expansions each with `1:` and `BNE ... 1b` assemble, and each branch targets its own loop.
- [ ] `1f` skips the current word's own `1:` and finds the next one.
- [ ] `1b` with no prior definition, and `1f` with no later one, raise `LabelError` with the location.
- [ ] A named label defined twice still fails with today's message.
- [ ] The `fold_32x32x144` digest is unchanged after conversion.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] Local labels are documented in the assembly-syntax reference, with one example.
- [ ] `check --json` reports the local-label errors with correct positions.
