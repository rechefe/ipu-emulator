# Adding Instructions

This guide shows how to add a new instruction to the IPU assembler and emulator.

## Overview

The IPU uses a **single source of truth** for all instruction definitions: `INSTRUCTION_SPEC` in `src/tools/ipu-common/src/ipu_common/instruction_spec.py`.

Adding an instruction is a 3-step process:

1. **Add to instruction_spec** — define the instruction with operands and documentation
2. **Implement execution handler** — write the Python method in the emulator's `Ipu` class  
3. **Add tests** — verify the instruction works correctly

No manual opcode management needed — opcodes are automatically derived from instruction position.

## Step 1: Add Instruction to `instruction_spec.py`

Open `src/tools/ipu-common/src/ipu_common/instruction_spec.py` and find the `INSTRUCTION_SPEC` dictionary. Locate the slot type where your instruction belongs (e.g., `"mult"`, `"acc"`, `"xmem"`, `"lr"`, `"cond"`, `"break"`).

Add your instruction to the slot's dictionary:

```python
INSTRUCTION_SPEC = {
    "mult": {
        # ... existing mult instructions ...
        
        "my_new_instruction": {
            "operands": [
                {"name": "dest", "type": "MultStageReg"},
                {"name": "src_a", "type": "MultStageReg", "read": "snapshot"},
                {"name": "src_b", "type": "MultStageReg", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="My New Operation",
                summary="Performs a custom operation on two source registers.",
                syntax="my_new_instruction Rd Ra Rb",
                operands=[
                    "Rd: Destination mult stage register (r0, r1, or mem_bypass)",
                    "Ra: First source mult stage register",
                    "Rb: Second source mult stage register",
                ],
                operation=(
                    "for i in [0, R_REG_SIZE):\n"
                    "    Rd[i] = custom_operation(Ra[i], Rb[i])"
                ),
                example="my_new_instruction r0 r1 mem_bypass;;",
            ),
            "execute_fn": "execute_my_new_instruction",
        },
    },
    # ... other slot types ...
}
```

**Key fields:**

- **`operands`**: List of operand definitions, each with:
  - `name`: Meaningful name for the operand (used in handler signature)
  - `type`: Operand type — one of:
    - `"MultStageReg"` — r0, r1, or mem_bypass
    - `"LrIdx"` — lr0-lr15
    - `"CrIdx"` — cr0-cr15
    - `"LcrIdx"` — lr0-lr15 or cr0-cr15
    - `"Immediate"` — 32-bit signed integer
    - `"BreakImmediate"` — 16-bit break condition
    - `"Label"` — branch target label
  - `read` (optional): `"snapshot"` or `"live"` — marks source registers whose values are auto-resolved:
    - `"snapshot"` — read from the VLIW snapshot (pre-write state, for read-before-write semantics)
    - `"live"` — read from current register file (sees writes from earlier slots)
    - Omit for destination registers or indices passed as raw values

- **`doc`**: `InstructionDoc` with:
  - `title` — Short human-readable name
  - `summary` — What the instruction does
  - `syntax` — Assembly syntax example
  - `operands` — Description of each operand (type and meaning)
  - `operation` — Pseudo-code showing the operation
  - `example` — Code example

- **`execute_fn`**: Name of the Python method in the `Ipu` class that executes this instruction (e.g., `"execute_my_new_instruction"`)

**Opcode assignment:**

Opcodes are **automatically derived from position** within the slot's instruction list. The first instruction gets opcode 0, the second gets opcode 1, etc. If you want to preserve existing opcodes, append new instructions to the end. To change opcodes, reorder instructions (both assembler and emulator will update automatically).

## Step 2: Implement Execution Handler

Open `src/tools/ipu-emu-py/src/ipu_emu/ipu.py` and find the section with handlers for your slot type (look for comments like `# MULT Instruction Handlers`).

Add your execution method to the `Ipu` class:

```python
class Ipu:
    # ... existing methods ...
    
    # -----------------------------------------------------------------------
    # MULT Instruction Handlers
    # -----------------------------------------------------------------------
    
    # ... existing mult handlers ...
    
    def execute_my_new_instruction(self, *, dest: int, src_a: bytearray,
                                    src_b: bytearray) -> None:
        """Execute my_new_instruction: performs custom operation.
        
        Args:
            dest: Destination register index (raw MultStageRegField value)
            src_a: Source A register bytes (auto-resolved from snapshot)
            src_b: Source B register bytes (auto-resolved from snapshot)
        """
        result = bytearray(R_REG_SIZE)
        dtype = DType(self.state.get_cr_dtype())
        
        for i in range(R_REG_SIZE):
            # Perform your custom operation
            result[i] = (src_a[i] + src_b[i]) & 0xFF  # Example: element-wise add
        
        # Write result to destination register
        reg_name, elem_idx = _MULT_STAGE_MAP[dest]
        self.state.regfile.set_register_bytes(reg_name, elem_idx, result)
```

**Key points:**

- **Method signature**: `def execute_<instruction_name>(self, *, <operand_names>) -> None:`
  - Use **keyword-only arguments** (note the `*` before operand names)
  - Parameter names **must match** the `name` fields in `instruction_spec`
  - Parameter order doesn't matter (they're passed as kwargs)

- **Auto-resolved operands** (those with `"read"` field):
  - Register operands with `"read": "snapshot"` receive `bytearray` values (register contents)
  - LR/CR operands with `"read"` receive `int` values (32-bit register value)
  - The dispatcher automatically extracts these from the snapshot or live regfile

- **Raw operands** (no `"read"` field):
  - Destination registers receive the raw index as `int`
  - Immediates receive the literal value as `int`
  - Labels receive the resolved address as `int`

- **Common patterns**:
  - Use `_MULT_STAGE_MAP[dest]` to get `(reg_name, elem_idx)` from a MultStageReg index
  - Use `self.state.regfile.set_register_bytes()` to write to a register
  - Use `DType(self.state.get_cr_dtype())` to get the current data type
  - Use `ipu_mult()`, `ipu_add()` from `ipu_emu.ipu_math` for typed arithmetic

**Available helpers:**

```python
# Register access (via self.state.regfile)
self.state.regfile.get_lr(idx) -> int          # Read LR register
self.state.regfile.set_lr(idx, value: int)     # Write LR register
self.state.regfile.get_r_acc_bytes() -> bytearray  # Read accumulator
self.state.regfile.set_r_acc_bytes(data: bytearray)  # Write accumulator

# Register metadata
reg_name, elem_idx = _MULT_STAGE_MAP[idx]      # Resolve MultStageReg index
R_REG_SIZE                                      # Register size in bytes (128)
R_ACC_SIZE                                      # Accumulator size (512)

# State access
self.state.xmem.read_address(addr, size) -> bytearray
self.state.xmem.write_address(addr, data: bytearray)
self.state.get_cr_dtype() -> int               # Get dtype from CR register
self.snapshot                                   # Pre-VLIW register snapshot

# Math operations
from ipu_emu.ipu_math import ipu_mult, ipu_add, DType
ipu_mult(a: int, b: int, dtype: DType) -> int
ipu_add(a: int, b: int, dtype: DType) -> int
```

## Step 3: Add Tests

Create tests in `src/tools/ipu-emu-py/test/test_<slot>_instructions.py` (or create a new file if needed):

```python
"""Tests for MULT instruction execution."""

import pytest
from ipu_emu.ipu_state import IpuState
from ipu_emu.ipu import Ipu

def test_my_new_instruction():
    """Test my_new_instruction performs custom operation correctly."""
    state = IpuState()
    ipu = Ipu(state)
    
    # Set up initial register values
    state.regfile.set_register_bytes("r", 0, bytearray(range(128)))
    state.regfile.set_register_bytes("r", 1, bytearray([1] * 128))
    
    # Create instruction word (see existing tests for encoding examples)
    inst = {
        "mult_inst_token_0_mult_inst_opcode": <opcode_value>,
        "mult_inst_token_1_mult_stage_reg_field": 0,  # dest = r0
        "mult_inst_token_2_mult_stage_reg_field": 0,  # src_a = r0
        "mult_inst_token_3_mult_stage_reg_field": 1,  # src_b = r1
        # ... other fields set to NOP opcodes ...
    }
    
    # Execute instruction
    state.inst_mem = [inst]
    state.regfile.set_pc(0)
    ipu.execute_vliw_cycle()
    
    # Verify result
    result = state.regfile.get_register_bytes("r", 0)
    expected = bytearray((i + 1) & 0xFF for i in range(128))
    assert result == expected
```

Run tests:

```bash
bazel test //src/tools/ipu-emu-py:test_execute
```

## Step 4: Verify Integration

After adding your instruction, verify it works in both assembler and emulator:

```bash
# Verify assembler accepts the instruction
echo "my_new_instruction r0 r1 mem_bypass;;" | bazel run //src/tools/ipu-as-py:ipu-as -- assemble --format hex -

# Run all tests
bazel test //...
```

## Summary

1. **Add to `instruction_spec.py`** — define operands, docs, and `execute_fn`
2. **Implement `execute_<name>` in `Ipu` class** — write the execution logic
3. **Add tests** — verify correctness with unit tests
4. **No opcode management** — opcodes are derived from position automatically

The instruction will automatically work in:
- Assembler (syntax highlighting, parsing, binary encoding)
- Emulator (execution, operand resolution, register updates)
- Documentation generation (help text, reference docs)

All from the single source of truth in `instruction_spec.py`.