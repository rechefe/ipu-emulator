# LR register file + 3-lane LR ALU RTL, verified against `execute_lr_*`

Part of the [CTRL-stage RTL epic](00-epic.md). Depends on #2. Can proceed in
parallel with #4.

## Goal

Implement `ctrl_lr_regfile.sv` (16×20-bit LR file) and `ctrl_lr_alu.sv`
(one ALU lane, instantiated ×3 per `LR_LANES` = 3), covering all six LR-slot
opcodes: `SET`, `ADD`, `SUB`, `INCR_MOD_POW2`, `INC`, `DEC`, plus `NOP`.

## Ground truth per opcode

Each bullet is the emulator handler that is the pass/fail oracle — spec
section is context, not the source of truth for exact arithmetic:

- **`SET`** — `execute_lr_set` (`ipu.py:511`): `dest = CR[src5[3:0]]` or
  sign-extended 4-bit immediate per `stage-control.md` §10.1.1's mode bit.
  Note the emulator's operand is a plain `CrIdx` (`"src"` resolved via
  `read: snapshot`) — the immediate-vs-CR mode-bit split described in the
  spec is an **encoding-level** detail belonging to the LR-slot union layout
  (`SLOT_UNIONS["lr"]`), not something `execute_lr_set` branches on in
  Python; confirm the RTL decode stage (not the ALU) is where that split
  belongs, matching how the assembler/emulator already split it.
- **`ADD`/`SUB`** — `execute_lr_add`/`execute_lr_sub` (`ipu.py:515-521`):
  plain wrap add/sub, no snapshot special-casing beyond the normal
  read-before-write rule (`src_a`/`src_b` are `"read": "snapshot"` operands
  in `instruction_spec.py`).
- **`INCR_MOD_POW2`** — `execute_lr_incr_mod_pow2` (`ipu.py:535-551`): reads
  `dest` from the **snapshot** (not live), adds `step`, then masks to
  `k_exp = k + LR_MOD_POW2_K_MIN` bits, where `k` is the *encoded* value
  (`k_semantic - 1`) — get `LR_MOD_POW2_K_MIN`/`LR_MOD_POW2_K_ENCODED_MAX`
  from `ipu_common/incr_mod_pow2_k.py` for the exact bounds, don't
  re-derive them from the spec prose.
- **`INC`/`DEC`** — `execute_lr_inc`/`execute_lr_dec` (`ipu.py:523-533`):
  explicit read-modify-write off the **snapshot** (`self.snapshot.get_lr`),
  not off any resolved operand — confirm the RTL lane reads its snapshot
  copy of `dest`, not a bypassed/live value, even though `dest` is also the
  write target this same cycle.
- **Register width** — all six write through `RegFile.set_scalar`, which
  masks to `LR_CR_SCALAR_VALUE_MASK` (20 bits, `ipu_config.py:24-25`)
  regardless of the `& 0xFFFFFFFF` masks visible in `ipu.py`'s handler
  bodies (those are vestigial/no-ops given the storage-layer mask). The RTL
  ALU should be 20-bit throughout — do not implement a 32-bit datapath just
  because the Python handler signatures look 32-bit-masked.

## Conflict detection

`Ipu._dispatch_lr_slots` (`ipu.py:557-593`) raises if two of the three lanes
in one VLIW word target the same destination LR. This is a same-cycle,
cross-lane hazard check, not a per-lane ALU behavior — implement it as
combinational cross-lane comparison logic in `ctrl_lr_regfile.sv` (or a
sibling conflict-detect block), not duplicated three times inside each ALU
lane. Decide the RTL's failure mode (the emulator raises a Python
exception, which has no synthesizable equivalent — likely an
assembler/decode-time invariant instead, since well-formed programs should
never encode this; confirm with #1 whether CTRL needs a runtime guard at
all or whether this is purely an assembler-side invariant).

## Tests (via the #2 parity harness)

- [ ] Each of the six opcodes, several operand combinations each, values
      spanning the 20-bit range including overflow/underflow wrap.
- [ ] `INCR_MOD_POW2` across the full legal `k` range
      (`LR_MOD_POW2_K_MIN` to the max), confirming the RTL mask matches the
      emulator's `((cur + step_u) & 0xFFFFFFFF) & mask` bit-for-bit after
      20-bit truncation.
- [ ] All three lanes active simultaneously with distinct destinations
      (parallel-lane correctness).
- [ ] Same-cycle conflicting destinations, once #1/this issue's conflict-mode
      decision is made, exercising whatever the agreed RTL behavior is.
- [ ] `NOP` lanes leave their would-be destination unchanged.

## Acceptance Criteria

- [ ] `ctrl_lr_regfile.sv` and `ctrl_lr_alu.sv` pass full parity coverage for
      all six opcodes against `execute_lr_*`.
- [ ] Register width is 20 bits throughout, matching `LR_CR_SCALAR_BITS`.
- [ ] Cross-lane same-destination conflict behavior is decided and
      implemented (or explicitly deferred to the assembler with a recorded
      rationale).
