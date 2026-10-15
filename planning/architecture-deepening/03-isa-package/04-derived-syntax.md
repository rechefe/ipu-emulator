# Derive `syntax` from mnemonic and operands; comma-free examples

Part of [Epic E03: ISA package](00-epic.md) · ID E03-04 · Size S · Depends on: E03-03 · Behaviour change: none for programs; generated docs and hover text change · `*_WIDTH` localparams: unchanged

## Goal

Remove the hand-written `syntax` line (Q15). It is derived as `MNEMONIC operand operand ...` from the instruction's name and operand names, so it can no longer disagree with the operands or with the grammar. Remove the commas from every `example`, so the published docs show code that assembles (CLAUDE.md: operands are separated by spaces). Fix the operand prose that the derived form exposes.

## Background

- `InstructionDoc.syntax` is hand-written (`instruction_spec.py:111-120`). 30 of the instruction syntax lines and 28 of the examples contain commas (34 and 32 with pseudo-instructions). `asm_grammar.lark` has no comma terminal (`ipu_as/asm_grammar.lark:15-21`), so `SET LR0, CR1;;` fails and `SET LR0 CR1;;` works.
- Mismatches today: `LDR_MULT_MASK_REG` syntax has a phantom `mask_idx` (`instruction_spec.py:251`); `INCR_MOD_POW2` syntax says `dst` (`:428`), the operand is `dest`.
- `gen_vscode._normalize_asm_snippet` (`gen_vscode.py:393-405`) rewrites commas to spaces in hover text; `gen_docs` publishes them verbatim. `test_hover_data.py` assembles every shipped example after that rewrite.
- Operand prose says `CR0–CR14` for the load/store `base` (`instruction_spec.py:236,254,290,327`); the register file and CLAUDE.md say CR0–CR15.
- The AGG.* summaries say "the active element count and partition come from `cr_idx`" (`instruction_spec.py:838-842` and the three siblings). The handlers use only `valid_elements` (`ipu_emu/ipu.py:1649-1703`); decision Q23 fixes the docs to match.

## Implementation

1. Remove `syntax` from `InstructionDoc` and from every spec entry (instructions and pseudo-instructions). Add `def syntax(mnemonic: str, operand_names: Sequence[str]) -> str` in `ipu_isa`, and use it in `gen_docs`, `gen_vscode` and anywhere else that read `doc.syntax` (E03-09 later exposes it as a property of the typed view).
2. Rewrite every `example` with spaces instead of commas (scripted; review the diff).
3. Add a schema rule (E03-03 rule list): no `,` in `example`.
4. Delete `_normalize_asm_snippet` and its call sites in `gen_vscode.py`; keep `test_code_fields_use_the_separator_the_grammar_accepts` in `test_hover_data.py` as the guard.
5. Prose fixes: `CR0–CR14` to `CR0–CR15` (4 places); AGG.* summaries say only `valid_elements` is read from `cr_idx` (4 places).

## Tests

- [ ] `test_hover_data.py` (every example assembles, no commas in code fields) and `test_isa_data.py` (`test_every_instruction_assembles_from_completion_values`) pass without the normalization step.
- [ ] New `ipu-isa/test/test_syntax.py`: `syntax("SET", ["reg", "src"]) == "SET reg src"`; every instruction's derived syntax names exactly its operands.
- [ ] E02 generated-artifact snapshots regenerated in this PR: `instructions.md` and `programmer-guide.md` lose commas in code and gain the prose fixes; `isa.json` is byte-identical (it was already normalized). The reviewer checks that the diff has no other change.
- [ ] E02 asm digests, decode snapshots and SV golden unchanged.

## Acceptance Criteria

- [ ] No `syntax=` remains in the spec; no example contains a comma.
- [ ] Generated docs show only syntax that assembles.
- [ ] Operand prose says CR0–CR15 and AGG docs say only `valid_elements` is used.
