# One handler module per slot; `Ipu` becomes orchestration

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-14 · Size M · Depends on: E04-12 · Behaviour change: none

## Goal

Split the 2351-line `ipu.py` into one module per slot (Q67). Each module holds that slot's `execute_*` handlers and the helpers only they use. `Ipu` keeps construction, plan building and the binding of handlers; the cycle lives in `cycle.py` and issue in `issue.py`. Files stay near the ~600-line guideline from the code-conventions page (Q66).

This is a move, not a rewrite: handler bodies are unchanged. E04-15 and E04-16 rewrite them against the codec.

## Background

- `ipu.py` holds plan building (ipu.py:246-530), operand resolution (ipu.py:795-833, 2044-2123), mask logic (ipu.py:838-990), load/store handlers (ipu.py:1021-1110), LR handlers (ipu.py:1113-1217), MULT (ipu.py:1219-1404), ACC and RESHAPE (ipu.py:1406-1615), AGG (ipu.py:1621-1703), AAQ and ST.AAQ (ipu.py:1705-1842), COND (ipu.py:1844-1883), BREAK (ipu.py:1887-1903), dispatch and the cycle (ipu.py:1905-2346).
- `_build_plan` binds each spec instruction to `Ipu.<execute_fn>` and raises if the method is missing (ipu.py:470-478, inside `_build_plan` at ipu.py:451). The CLAUDE.md rule "operand names in `execute_*` handlers must exactly match the spec" is enforced through these signatures.

## Implementation

1. Create `ipu_emu/slots/` with `lr.py`, `load.py`, `mult.py`, `acc.py` (ACC, AGG, RESHAPE), `aaq.py`, `store.py` (store and `acc_store`), `cond.py`, `brk.py` (`break` is a keyword). Each module defines plain functions `execute_<name>(ipu, *, ...)` with the same keyword-only operand names.
2. `_build_plan` looks the handler up in the slot's module (`slots.<slot>.execute_<name>`), not on `Ipu`. The "handler missing" check and the operand-name check stay, now against module functions. Update CLAUDE.md's checklist step 2 and `docs/content/adding-instruction.md` to say "add `execute_<name>` to `ipu_emu/slots/<slot>.py`".
3. Shared helpers (mask, LRD byte access, row address) move to `geometry.py` (E04-11) or stay as small private functions in the one slot module that uses them.
4. `Ipu` keeps: `__init__`, state references, the bound plan tables, and delegates required by the debuggers until E08.
5. Move in several commits (one slot per commit) so review diffs are pure moves (`git diff --color-moved`).

## Tests

- [ ] Output-identical proof: E02 kernel output digests and timing characterization tests; full `bazel test //...` and `--define=ipu_proto=1`.
- [ ] Existing spec/handler consistency test (operand names match handler keyword-only parameters) still runs, now against slot modules; add one if none exists.
- [ ] E04-02 benchmark unchanged within noise.

## Acceptance Criteria

- [ ] `ipu.py` holds no `execute_*` handler.
- [ ] No module in `ipu_emu/slots/` imports another slot module.
- [ ] Every file in `ipu_emu/` is under 800 lines; slot modules aim for 600.
- [ ] CLAUDE.md and `adding-instruction.md` describe the new location.
