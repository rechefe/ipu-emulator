# Explanation: timing model

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-18 · Size S · Depends on: E09-06, E03 (declared timing contract in `ipu_isa`), E04 (cycle view and timing behaviour PRs, including the 3 `attn_v_*` kernel fixes) · Behaviour change: none

## Goal

A kernel author knows, for every slot, which register values it reads in a
given cycle, so they can write correct same-word code without reading the
emulator.

## Background

- The current docs state one rule: "Slots see a register snapshot at cycle
  start (read-before-write semantics)" (`SKILL.md:76`;
  `adding-instruction.md:73-74` describes `"read": "snapshot"` and `"live"`).
  The real behaviour already has exceptions:
  `isa-alias-catalogue.md:271-273` (`MULT.RC.*` reads `R_CYCLIC` from the
  start-of-cycle snapshot), and requirement OQ-9 and OQ-10 discuss same-word
  load/use and LR forwarding.
- Q32 fixes the contract:
  1. the Control stage (LR, COND) reads LRs at cycle start; all other stages
     read LRs after this word's LR writes;
  2. MULT reads R0, R1, R_CYCLIC and R_MASK at cycle start (Q20);
  3. `MULT_RES` reaches ACC, AGG and RESHAPE in the same cycle (Q19), and
     `R_ACC` reaches AAQ in the same cycle (Q21);
  4. LR sub-slots read cycle-start values.
- E03 declares this contract as data in `ipu_isa`; E04 implements it and fixes
  three `attn_v_*` kernels in a behaviour PR.
- Q18: the user considers it a problem that stages outside Control used
  snapshot LR values; rule 1 is the answer.

## Implementation

1. Write `explanation/timing-model.md` with `anthropic-skills:technical-writing`:
   - The model in one paragraph: one word retires per cycle; within a word,
     some values are read at cycle start and some see same-word results.
   - A table generated from the declared timing contract in `ipu_isa`: slot,
     operand or register, read point (cycle start, after LR writes, same-cycle
     forward). Generated so it cannot drift from the emulator.
   - One assembling example per rule, showing the value each slot sees.
   - What the emulator does not model (stalls, F-8; link OQ-28) and how this
     relates to the hardware pipeline (link `requirements/architecture.md`
     and the control stage spec, without changing them).
   - When a kernel needs an extra word, and why (link E09-10 idioms).
2. Remove the "snapshot at cycle start" sentence from `SKILL.md` and
   `CLAUDE.md`, and link this page.
3. Point the generated instruction reference (E09-14) at this page from each
   operand's read point.
4. If requirement rows OQ-9 or OQ-10 are answered by Q32, tell the requirement
   owners; do not edit the requirement pages in this issue.

## Tests

- [ ] `bazel test //docs:test_doc_asm_blocks` passes; every example assembles.
- [ ] Each example is also run as an emulator test that asserts the value the
      page claims (included from the test file, so page and test agree).

## Acceptance Criteria

- [ ] Every rule of the timing contract has a generated table row and a tested
      example.
- [ ] No page states the old "all slots read a cycle-start snapshot" rule.
