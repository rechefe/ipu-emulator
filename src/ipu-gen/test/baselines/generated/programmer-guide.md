# Programmer's Guide: Pseudo-Instructions

Pseudo-instructions are assembly mnemonics that the assembler expands into a real instruction at compile time. They never get an opcode and never appear in the binary, so they cost nothing at runtime — see [Adding a pseudo-instruction](../../how-to/add-an-instruction.md#pseudo-instructions) for how they're declared. This page is **generated** from `PSEUDO_INSTRUCTION_SPEC` in `instruction_spec.py`.

### `BGT.S` — Branch if Greater Than (pseudo)

**Syntax:** `BGT.S reg1 reg2 label`

**Tag:** `ASSEMBLER` (expanded by the assembler; no opcode)

**Faults:** F-2, through `BLT.S` ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg1: First register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `reg2` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg2: Second register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if first register is greater than second.

> Expands to `BLT.S reg2 reg1 label` at assemble time (operands swapped). Identical encoding and runtime cost to a hand-written BLT.S.

**Pseudo code:**
`if (reg1 > reg2) PC = label`

**Example of usage:**
```asm
BGT.S LR0 LR1 bigger;;
bigger:
HALT;;
```

**Expands to:** `BLT.S reg2 reg1 label` (cond slot)

---

### `BLE.S` — Branch if Less or Equal (pseudo)

**Syntax:** `BLE.S reg1 reg2 label`

**Tag:** `ASSEMBLER` (expanded by the assembler; no opcode)

**Faults:** F-2, through `BGE.S` ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg1: First register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `reg2` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg2: Second register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if first register is less than or equal to second.

> Expands to `BGE.S reg2 reg1 label` at assemble time (operands swapped), exactly mirroring how BGT.S expands to BLT.S. Identical encoding and runtime cost to a hand-written BGE.S.

**Pseudo code:**
`if (reg1 <= reg2) PC = label`

**Example of usage:**
```asm
BLE.S LR0 LR1 smaller_or_equal;;
smaller_or_equal:
HALT;;
```

**Expands to:** `BGE.S reg2 reg1 label` (cond slot)

---

### `BZ` — Branch if Zero (pseudo)

**Syntax:** `BZ reg label`

**Tag:** `ASSEMBLER` (expanded by the assembler; no opcode)

**Faults:** F-2, through `BEQ` ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg: Register to test (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if register is zero (CR0 is the constant 0).

> Expands to `BEQ reg CR0 label`; CR0 is the read-only constant 0.

**Pseudo code:**
`if (reg == 0) PC = label`

**Example of usage:**
```asm
BZ LR0 done;;
done:
HALT;;
```

**Expands to:** `BEQ reg CR0 label` (cond slot)

---

### `BNZ` — Branch if Not Zero (pseudo)

**Syntax:** `BNZ reg label`

**Tag:** `ASSEMBLER` (expanded by the assembler; no opcode)

**Faults:** F-2, through `BNE` ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg: Register to test (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if register is not zero (CR0 is the constant 0).

> Expands to `BNE reg CR0 label`; CR0 is the read-only constant 0.

**Pseudo code:**
`if (reg != 0) PC = label`

**Example of usage:**
```asm
loop:
DEC LR0 1;;
BNZ LR0 loop;;
```

**Expands to:** `BNE reg CR0 label` (cond slot)

---

### `B` — Unconditional Branch (pseudo)

**Syntax:** `B label`

**Tag:** `ASSEMBLER` (expanded by the assembler; no opcode)

**Faults:** F-2, through `BEQ` ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Always branch to label.

> Expands to `BEQ CR0 CR0 label`. CR0 always equals itself, so the branch is always taken.

**Pseudo code:**
`PC = label`

**Example of usage:**
```asm
start:
INC LR0 1;;
B start;;
```

**Expands to:** `BEQ CR0 CR0 label` (cond slot)

---
