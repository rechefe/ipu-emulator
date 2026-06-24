# Programmer's Guide: Pseudo-Instructions

Pseudo-instructions are assembly mnemonics that the assembler expands into a real instruction at compile time. They never get an opcode and never appear in the binary, so they cost nothing at runtime — see [Adding a pseudo-instruction](adding-instruction.md#adding-a-pseudo-instruction) for how they're declared. This page is **generated** from `PSEUDO_INSTRUCTION_SPEC` in `instruction_spec.py`.

### `BGT` — Branch if Greater Than (pseudo)

**Syntax:** `BGT reg1, reg2, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR14) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if first register is greater than second.

> Expands to `BLT reg2, reg1, label` at assemble time (operands swapped). Identical encoding and runtime cost to a hand-written BLT.

**Pseudo code:**
`if (reg1 > reg2) PC = label`

**Example of usage:**
```asm
BGT LR0, LR1, bigger;;
```

**Expands to:** `BLT` (cond slot), arguments `reg2, reg1, label`

---

### `BLE` — Branch if Less or Equal (pseudo)

**Syntax:** `BLE reg1, reg2, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR14) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if first register is less than or equal to second.

> Expands to `BGE reg2, reg1, label` at assemble time (operands swapped), exactly mirroring how BGT expands to BLT. Identical encoding and runtime cost to a hand-written BGE.

**Pseudo code:**
`if (reg1 <= reg2) PC = label`

**Example of usage:**
```asm
BLE LR0, LR1, smaller_or_equal;;
```

**Expands to:** `BGE` (cond slot), arguments `reg2, reg1, label`

---

### `BZ` — Branch if Zero (pseudo)

**Syntax:** `BZ reg, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LcrIdx`](operand-types.md#lcridx) | reg: Register to test (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if register is zero. Assumes CR0 always holds zero.

> Expands to `BEQ reg, CR0, label`. Assumes CR0 always holds 0.

**Pseudo code:**
`if (reg == 0) PC = label`

**Example of usage:**
```asm
BZ LR0, done;;
```

**Expands to:** `BEQ` (cond slot), arguments `reg, CR0, label`

---

### `BNZ` — Branch if Not Zero (pseudo)

**Syntax:** `BNZ reg, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LcrIdx`](operand-types.md#lcridx) | reg: Register to test (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if register is not zero. Assumes CR0 always holds zero.

> Expands to `BNE reg, CR0, label`. Assumes CR0 always holds 0.

**Pseudo code:**
`if (reg != 0) PC = label`

**Example of usage:**
```asm
BNZ LR0, loop;;
```

**Expands to:** `BNE` (cond slot), arguments `reg, CR0, label`

---

### `B` — Unconditional Branch (pseudo)

**Syntax:** `B label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Always branch to label.

> Expands to `BEQ CR0, CR0, label`. CR0 always equals itself, so the branch is always taken.

**Pseudo code:**
`PC = label`

**Example of usage:**
```asm
B start;;
```

**Expands to:** `BEQ` (cond slot), arguments `CR0, CR0, label`

---
