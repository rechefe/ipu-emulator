# Dispatch-bus packing, XMEM read-address path, and BREAK/BKPT priority gating

Part of the [CTRL-stage RTL epic](00-epic.md). Depends on #3 and #4.

## Goal

Implement `ctrl_dispatch.sv`: packing of `mult_vliw_bus`/`acc_vliw_bus`/
`aaq_vliw_bus`/`str_vliw_bus`, the `xmem_read_addr`/`xmem_read_en` resolution,
and the same-cycle BREAK-gates-everything-else priority logic from #1.

## Bus packing and operand timing

Ground truth: `stage-control.md` §5, §7 — this is the one area where the
spec is detailed and precise and the emulator's dispatch order
(`Ipu.execute_vliw_cycle`, `ipu.py:1213-1252`) independently confirms the
same ordering, so cross-check both:

- CTRL evaluates the 3 LR-ALU lanes **before** packing any dispatch bus, so
  every bus carries **post-LR-write** CR/LR operand values for this cycle —
  not the snapshot. Confirm against the emulator's actual call order:
  `_dispatch_lr_slots` runs before `dispatch_instruction("load"/"mult"/...)`
  in `execute_vliw_cycle`, and `dispatch_instruction` resolves each
  instruction's `"read"` operands (`"live"` vs `"snapshot"`) via
  `self._resolve_operand(..., self.state.regfile if source == "live" else
  self.snapshot)` — i.e. most MULT/ACC/AAQ/STORE operands are declared
  `"read": "live"` in `instruction_spec.py` precisely because they must see
  this cycle's LR writes. Any operand marked `"read": "snapshot"` in a
  downstream slot is the exception to flag and check individually against
  §5's stated rule.
- CTRL's **own** reads (LR-ALU sources, cond-slot operands) use the
  **prior-cycle snapshot** — the opposite timing from what it forwards.
  This asymmetry is the single most important timing fact in the spec; get
  a cocotb test that would fail if the two were accidentally swapped (e.g.
  an LR write this cycle that must NOT be visible to this cycle's own
  branch condition, but MUST be visible on the forwarded bus to MULT).

## XMEM read-address path

- `xmem_read_addr = CR[load.base_idx] + LR[load.offset_idx]`, using the
  **post-LR-write** LR value (§7.1) — same timing rule as the dispatch
  buses, not the snapshot.
- `xmem_read_en` asserted iff the load slot is non-NOP and no bubble is
  active. Cross-check against `LDR_MULT_REG`/`LDR_CYCLIC_MULT_REG`/
  `LDR_MULT_MASK_REG`'s operands in `instruction_spec.py` — all three
  declare `offset`/`base` as `"read": "live"`, consistent with post-write
  timing.
- Per §9, there are currently **no CTRL-level hazard interlocks** — the
  AAQ-scalar-register RAW hazard that used to require bubble insertion was
  removed when cross-lane aggregation moved into the ACC slot. Confirm the
  RTL has no leftover bubble-insertion logic for a hazard that no longer
  exists; `bubble` in the `xmem_read_en` expression above should currently
  always evaluate false, not dead logic guarding against a hazard the ISA
  no longer has.

## BREAK/BKPT priority gating

Per #1's resolution. At minimum, this block must reproduce the emulator's
same-cycle ordering guarantee: `dispatch_instruction("break", inst)` runs
**first**, and a `BREAK`/`BREAK.IFEQ` hit **prevents every other slot's side
effects this cycle** — LR writes, dispatch-bus sends, and the
`xmem_read_addr`/`en` strobe all become no-ops for that cycle
(`ipu.py:1237-1240`, contrast with `execute_vliw_cycle_skip_break` at
`ipu.py:1254-1278`, used only once a debugger has already stepped past a
break). This is a real combinational gate on every write-enable and
bus-valid signal in the stage, not just a status flag that happens to be
set — build it as such.

## Tests (via the #2 parity harness)

- [ ] For each MULT/ACC/AAQ/STORE-slot instruction's `"read"` operands
      (sampled across at least one instruction per slot, not exhaustively —
      the ISA-level correctness of those slots is #3/#4's problem, not
      this issue's), confirm the packed bus value matches the emulator's
      resolved operand exactly, with an LR write happening the same cycle to
      prove the post-write timing.
- [ ] A cond-slot branch reading an LR that an LR-ALU lane writes the same
      cycle: confirm the branch's own comparison uses the **pre**-write
      snapshot value while the same cycle's dispatch buses use the
      **post**-write value — both from a single VLIW word.
- [ ] `xmem_read_addr`/`xmem_read_en` for each of the three load
      instructions, NOP (deasserted), and a same-cycle LR write feeding the
      offset.
- [ ] BREAK/BREAK.IFEQ (both taken and not-taken) alongside a same-cycle LR
      write and dispatch instruction: confirm all downstream side effects
      are suppressed exactly when `BreakResult.BREAK` fires, and occur
      normally otherwise.

## Acceptance Criteria

- [ ] All four dispatch buses and the XMEM read-address path match the
      post-LR-write / pre-write-snapshot asymmetry exactly, per instruction
      operand sampled.
- [ ] No dead hazard/bubble logic remains for the removed AAQ RAW hazard.
- [ ] BREAK/BKPT gating suppresses every other slot's side effects
      same-cycle, matching `execute_vliw_cycle`'s early-return behavior.
