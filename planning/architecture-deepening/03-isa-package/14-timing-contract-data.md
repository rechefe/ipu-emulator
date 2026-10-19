# Declare the timing contract as data

Part of [Epic E03: ISA package](00-epic.md) · ID E03-14 · Size M · Depends on: E03-09, E02-05, E02-06 · Behaviour change: none (declares data; emulator changes are E04-06, E04-12, E04-25 to E04-27) · `*_WIDTH` localparams: unchanged (new keys are not operand fields; checked by E02-03)

## Goal

Make the rule for which value a read sees within one cycle part of the ISA (Q31, Q32), instead of a side effect of call order in `ipu.py`. `ipu_isa` declares:

- the slot execution order;
- the Q32 contract as four rules;
- per instruction, every register it reads or writes, including the ones that today are read inside handlers with no declaration;
- the places where the emulator today differs from the contract, as a pinned table that can only shrink.

E04 then consumes this data (E04-06 slot order, E04-12 cycle view) and fixes each deviation in its own behaviour PR (E04-25, E04-26, E04-27).

## Background

- Slot order exists only as code: `execute_vliw_cycle` (`ipu_emu/ipu.py:2297-2304`), `execute_vliw_cycle_skip_break` (`:2336-2343`), `_SLOT_ORDER` (`:2134`), and prose in `gen_docs.py:200`.
- `read: snapshot | live` on operands (`instruction_spec.py` module docstring, lines 17-22, 61-72) is the only declared timing. Handlers read other registers directly: R_MASK (`ipu.py:945`), MULT_RES (`:1256-1500`, `:1603`, `:1653-1698`), R_ACC (`:1421`, `:1448`, `:1476`, `:1668`, `:1701`, `:1765`), POST_AAQ_REG (`:1838`), MULT.RC.VE `src` (`:1234`), ACC.RESHAPE `reshape_mask` LR (`:1594`), INC/DEC/ADDB/ADDBI destination LRs (`:1128-1167`).
- The LR write target is found by operand name: `op["name"] in ("dest", "reg") and "read" not in op` (`ipu.py:466-470`). Renaming an operand would silently disable the LR write-conflict check.
- Decision Q32:
  1. the Control stage (LR, COND) reads LRs at cycle start; all other stages read the value after the LR writes of the same word;
  2. MULT reads R0, R1, R_CYCLIC and R_MASK at cycle start;
  3. MULT_RES is forwarded to ACC, AGG and RESHAPE; R_ACC is forwarded to AAQ;
  4. LR sub-slots read cycle-start values.
- E02-05 and E02-06 pin today's behaviour row by row. Five rows differ from Q32: AGG.* `dest_slot`, ACC.RESHAPE `source` and `dest` (rule 1), ACC.RESHAPE MULT_RES (rule 3), MULT R_MASK (rule 2).
- GLOSSARY.md: use "cycle start" and "forwarded" in prose, not "snapshot" and "live".

## Implementation

1. `ipu_isa/timing.py`:
   ```python
   SLOT_EXECUTION_ORDER: tuple[Slot, ...] = (BREAK, LR, LOAD, MULT, ACC, AAQ, STORE, ACC_STORE, COND)
   CONTROL_SLOTS: frozenset[Slot] = {LR, COND, BREAK}      # BREAK is evaluated before any write

   class View(StrEnum): CYCLE_START = "cycle_start"; FORWARDED = "forwarded"

   @dataclass(frozen=True, slots=True)
   class RegisterAccess:
       register: str                 # REGISTER_DEFINITIONS name: "lr", "r", "r_cyclic", "r_mask", "mult_res", "r_acc", "post_aaq_reg"
       via_operand: str | None       # operand that selects the index, e.g. "dest_slot"; None for whole registers
       view: View | None             # reads only

   def contract_view(slot: Slot, register: str) -> View   # Q32 rules 1-4; anything else: FORWARDED if a producer precedes the slot in SLOT_EXECUTION_ORDER
   ```
2. Spec schema additions (E03-03 key tables):
   - operand key `write: true` on destination register operands (`SET reg`, `ADD dest`, `INC dest`, `ADDB dest`, `LDR_MULT_REG dest`, ...);
   - instruction keys `implicit_reads` and `implicit_writes`, lists of `RegisterAccess` dicts;
   - `read` values renamed `snapshot` to `cycle_start` and `live` to `forwarded` in the spec, in `Read` (E03-09) and in the emulator's single comparison site (mechanical; no alias).
3. Fill the data with today's behaviour, instruction by instruction, using the E02-05/E02-06 tests as the reference. Example: `ACC.ADD` gets `implicit_reads: [{register: "mult_res", view: "forwarded"}, {register: "r_acc", view: "cycle_start"}]`, `implicit_writes: [{register: "r_acc"}]`.
4. `KNOWN_DEVIATIONS` in `timing.py`: one entry per place where the declared (today's) view differs from `contract_view`, each naming the E04 issue that fixes it:

   | Instruction | Read | Today | Contract | Fixed by |
   |---|---|---|---|---|
   | AGG.SUM(.FIRST), AGG.MAX(.FIRST) | LR via `dest_slot` | cycle start | forwarded | E04-25 |
   | ACC.RESHAPE | LR pairs via `source`, `dest` | cycle start | forwarded | E04-25 |
   | ACC.RESHAPE | MULT_RES | cycle start | forwarded | E04-26 |
   | MULT.* | R_MASK | forwarded | cycle start | E04-27 |

   The validator computes the actual differences and requires them to equal this table exactly: a new difference fails, and a fixed one fails until its row is deleted.
5. Validator rules (run by `load_isa`):
   - a forwarded read of a register is legal only if every slot that writes it (from `write` operands and `implicit_writes`) precedes the reader in `SLOT_EXECUTION_ORDER`; a same-slot producer must be read at cycle start;
   - CR reads are cycle start (nothing writes a CR during a run);
   - every read in a Control slot is cycle start;
   - every LR-slot instruction that writes an LR declares it with `write: true`.
6. Emulator: replace the name convention at `ipu.py:466-470` with the declared `write` flag. This is the only emulator edit; the E02 LR-conflict tests prove it identical.
7. Docs: `gen_docs` renders the execution order, the four rules and a per-instruction read/write table from this data, plus the deviation table, replacing the prose at `gen_docs.py:200`. Fix the ACTIVATE.QUANTIZE summary ("r_acc (snapshot)", `instruction_spec.py:1002`) to "forwarded R_ACC", which matches the code and the contract.

## Tests

- [ ] `ipu-isa/test/test_timing.py`: `contract_view` for each Q32 rule; each validator rule fails on a planted violation (forwarded read from a later slot, forwarded CR read, a Control-slot forwarded read, a new undeclared deviation, a stale deviation row).
- [ ] The declared views equal the E02-05/E02-06 pinned behaviour: a generated test maps each timing test to the declaration it exercises.
- [ ] E02 baselines unchanged except the generated docs pages (new tables, ACTIVATE summary) and `isa.json` hover text.

## Acceptance Criteria

- [ ] Slot execution order, the Q32 rules and every register access are data in `ipu_isa`.
- [ ] Every difference between today's emulator and the contract is listed in `KNOWN_DEVIATIONS` with the E04 issue that removes it.
- [ ] The LR write target is declared, not inferred from operand names.

## Risks

- Implicit reads are filled in by reading handler code; a missed read is not caught until E04-12 makes undeclared reads fail. Mitigation: the E02-05/E02-06 tests cover every register in the table; E04-12's generated conformance test closes the gap.
