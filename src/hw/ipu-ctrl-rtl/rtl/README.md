# CTRL-stage RTL — status

This is a first-look scaffold for
[`planning/ctrl-stage-rtl/`](../../../planning/ctrl-stage-rtl/), written to
show what the CTRL stage's RTL looks like in practice, not a finished
deliverable. **It has never been run through a simulator** — no Verilator/
Icarus is installed in this environment, so nothing here has been
elaborated or simulated yet. Standing up that toolchain under Bazel is
[issue #2](../../../planning/ctrl-stage-rtl/02-rtl-scaffold-verification-infra.md);
until it lands, treat this as "read it and check the logic by eye against
`ipu.py`," not "it compiles, therefore it's right."

## What's real

- **`ctrl_pkg.sv`** — parameters and opcode enums, all derived by directly
  evaluating `ipu_common.union_layout.compute_slot_layout()` /
  `SLOT_UNIONS` for the `lr` and `cond` slots (not eyeballed from the spec
  prose — see the two findings below, which only came from doing that).
- **`ctrl_lr_lane.sv`** — one LR-slot ALU lane (`SET`, `ADD`, `SUB`,
  `INCR_MOD_POW2`, `INC`, `DEC`, `NOP`), matching `ipu.py`'s
  `execute_lr_*` handlers instruction-by-instruction.
- **`ctrl_lr_regfile.sv`** — the 16×20-bit LR file, 3 write ports, a
  same-destination conflict flag.
- **`ctrl_lr_stage.sv`** — wires 3× `ctrl_lr_lane` + `ctrl_lr_regfile`
  together; this is the full deliverable for
  [issue #3](../../../planning/ctrl-stage-rtl/03-lr-regfile-alu.md).

This covers the LR slot end to end — every LR opcode, all three lanes, both
the pre-write ("snapshot") and post-write ("forwarded") read views the rest
of the stage needs.

## What's a stub

- **`ctrl_stage.sv`** — the top-level port list from `stage-control.md`
  section 3.0, with `ctrl_lr_stage` wired in for real and every other block
  (fetch/PC/`inst mem`/`inst $`, cond resolver, CR file, dispatch-bus
  packing, XMEM address path, APB slave) tied to an inert default and
  marked `TODO(#N)` against the owning planning issue. Its `lr_inst`/
  `cr_rdata` inputs are tied to NOP/zero rather than connected to a real
  fetch path, because that path doesn't exist yet.
- Nothing for `BEQ`/`BNE`/`BLT`/`BGE`/`BR`/`BKPT`, the CR file, the APB
  slave, or dispatch-bus packing exists yet — those are issues #4–#6.

## Two things this scaffold found that the planning docs didn't have yet

Both came from actually running the union-layout solver
(`ipu_common.union_layout.compute_slot_layout`) instead of reading
`stage-control.md`'s prose at face value — worth remembering as a general
lesson for the rest of this epic, not just these two spots:

1. **`SET` has no immediate mode.** `stage-control.md` section 10.1.1
   describes a 5-bit `src5` operand with a mode-select MSB choosing between
   a CR read and a sign-extended 4-bit immediate. The real union layout
   gives `SET` a plain `CrIdx` operand (`instruction_spec.py`), and
   `execute_lr_set` (`ipu.py:511`) has no immediate branch — it's
   unconditionally `reg = CR[src]`. The assembler's own generated docs
   (`ipu_as/gen_docs.py:200`) agree: "SET copies from a cr register."
   `ctrl_pkg.sv`/`ctrl_lr_lane.sv` implement the real, CR-only behavior.
   This needs a `stage-control.md` correction — see
   [issue #1](../../../planning/ctrl-stage-rtl/01-spec-emulator-reconciliation.md).
2. **More evidence for the IMEM-depth discrepancy.** The cond slot's
   `Label` operand is **10 bits** wide (`get_operand_type_bits()`:
   `"Label": 10`, sized for a 1024-entry program). That's the emulator's
   `INST_MEM_SIZE = 1024`, not the spec's hardware `IMEM_DEPTH = 256`
   (8 bits) or `PC_W = 7` (128/bank). The *binary encoding itself* — not
   just the emulator's array size — is built for the flat 1024-word
   address space. Recorded in `ctrl_pkg.sv`'s `COND_LABEL_W` comment and
   folded into issue #1's gap #3.

## Next steps

Follow `planning/ctrl-stage-rtl/00-epic.md`'s dependency order: issue #1
(spec reconciliation, including the `SET` finding above) blocks the rest;
issue #2 (toolchain) is what turns "read it by eye" into "simulated and
checked against `ipu_emu.Ipu`."
