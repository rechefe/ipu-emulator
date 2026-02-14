# IPU Instruction Specification Refactoring — COMPLETED ✅

## Executive Summary

Successfully refactored the IPU instruction system to address three user-identified problems:

1. **No Explicit Opcodes**: Opcodes are now position-based and auto-derived from instruction position in the spec
2. **No Code Generation**: Eliminated file-based codegen; all classes and constants created at runtime via factories
3. **Meaningful Operands**: Operands are now structured objects with names and semantic types

**Result**: Single source of truth in `INSTRUCTION_SPEC` → both assembler and emulator automatically stay in sync.

---

## Problems Addressed

### Problem 1: Explicit Opcodes (SOLVED)
**Before**: 
```python
"str_acc_reg": {
    "opcode": 0,  # ← Manual, easy to get wrong
    "operands": ["CrIdx", "CrIdx"],
    ...
}
```

**After**:
```python
"str_acc_reg": {
    "operands": [
        {"name": "offset", "type": "CrIdx"},
        {"name": "base", "type": "CrIdx"},
    ],
    ...
}
# opcode = position in dict (position 0 → opcode 0)
```

**Benefit**: No manual opcode assignment; position = opcode automatically.

---

### Problem 2: Code Generation (SOLVED)
**Before**:
- Manual `codegen.py` script that wrote `opcodes_generated.py` files
- Assembly step required to use opcodes
- Generated files could get out of sync

**After**:
- Runtime factory functions `create_assembler_opcodes()` and `create_emulator_constants()`
- Opcodes/constants generated on-the-fly from `INSTRUCTION_SPEC`
- Always in sync by design

**Files deleted**:
- `src/tools/ipu-common/src/ipu_common/codegen.py` (no longer needed)

---

### Problem 3: Unstructured Operands (SOLVED)
**Before**:
```python
"operands": ["CrIdx", "CrIdx"]  # Generic list, unclear what each means
```

**After**:
```python
"operands": [
    {"name": "offset", "type": "CrIdx"},  # Named, typed
    {"name": "base", "type": "CrIdx"},
]

# Query:
operand_list = get_operand_names_and_types("xmem", "str_acc_reg")
# Returns: [("offset", "CrIdx"), ("base", "CrIdx")]
```

**Benefit**: Clear semantic meaning; tools can correlate names to parsed arguments.

---

## Architecture Changes

### 1. Single Source of Truth: `INSTRUCTION_SPEC`

**File**: `src/tools/ipu-common/src/ipu_common/instruction_spec.py`

Structure (position-based):
```python
INSTRUCTION_SPEC = {
    "xmem": {
        "str_acc_reg": {
            "operands": [...],      # Position 0 → opcode 0
            "doc": InstructionDoc(...),
            "execute_fn": "execute_str_acc_reg",
        },
        "ldr_mult_reg": { ... },    # Position 1 → opcode 1
        # ... all xmem instructions in order
    },
    "lr": { ... },
    "mult": { ... },
    "acc": { ... },
    "cond": { ... },
    "break": { ... },
}
```

**Key Properties**:
- Dict preserves insertion order (Python 3.7+)
- FIRST instruction in slot = opcode 0
- SECOND instruction in slot = opcode 1
- No explicit opcode fields needed

---

### 2. Runtime Opcode Factories

#### Assembler Opcodes Factory
```python
from ipu_common.instruction_spec import create_assembler_opcodes

opcodes = create_assembler_opcodes()
# Returns:
# {
#     "XmemInstOpcode": <class>,
#     "LrInstOpcode": <class>,
#     "MultInstOpcode": <class>,
#     "AccInstOpcode": <class>,
#     "CondInstOpcode": <class>,
#     "BreakInstOpcode": <class>,
# }

# Usage:
XmemInstOpcode = opcodes["XmemInstOpcode"]
XmemInstOpcode.enum_array()
# Returns: ["str_acc_reg", "ldr_mult_reg", "ldr_cyclic_mult_reg", ...]
```

**How it works**:
1. Dynamically creates Opcode subclasses at runtime
2. Each class's `enum_array()` returns instruction names in position order
3. Position in list = opcode index
4. No file writing, no codegen pipeline

---

#### Emulator Constants Factory
```python
from ipu_common.instruction_spec import create_emulator_constants

constants = create_emulator_constants()
# Returns:
# {
#     "XMEM_OP_STR_ACC_REG": 0,
#     "XMEM_OP_LDR_MULT_REG": 1,
#     ...
#     "LR_OP_INCR": 0,
#     "LR_OP_SET": 1,
#     ...
#     "NUM_XMEM_OP": 5,
#     "NUM_LR_OP": 4,
#     ...
# }
```

**How it works**:
1. Iterates through INSTRUCTION_SPEC
2. For each instruction, generates constants like `SLOT_OP_NAME = position`
3. Also generates `NUM_SLOT_OP` = total count
4. Updated at runtime, always matches spec

---

### 3. Updated Module Imports

#### `ipu-as-py/src/ipu_as/opcodes.py`
```python
# OLD: 27 lines of manual class definitions
# NEW:
from ipu_common.instruction_spec import create_assembler_opcodes

_opcode_classes = create_assembler_opcodes()
XmemInstOpcode = _opcode_classes["XmemInstOpcode"]
LrInstOpcode = _opcode_classes["LrInstOpcode"]
# ... etc (exports for backward compatibility)
```

**Benefits**:
- Source is now single instruction_spec
- Assembler imports never go stale

---

#### `ipu-emu-py/src/ipu_emu/execute.py`
```python
# OLD: 50+ hardcoded constants
# NEW:
from ipu_common.instruction_spec import create_emulator_constants

_emu_constants = create_emulator_constants()
XMEM_OP_STR_ACC_REG = _emu_constants["XMEM_OP_STR_ACC_REG"]
# ... (backward-compatible naming)
```

**Benefits**:
- Emulator constants always match assembler opcodes
- Position-based = no sync issues

---

## Operand Type Reference

All operand types are defined as string names in the spec, resolved by ipu_as:

| Type Name | Maps To | Legal Values |
|-----------|---------|--------------|
| `MultStageReg` | `ipu_as.reg.MultStageRegField` | r0, r1, mem_bypass |
| `LrIdx` | `ipu_as.reg.LrRegField` | lr0-lr15 |
| `CrIdx` | `ipu_as.reg.CrRegField` | cr0-cr15 |
| `LcrIdx` | `ipu_as.reg.LcrRegField` | lr0-lr15, cr0-cr15 |
| `Immediate` | `ipu_as.immediate.LrImmediateType` | 32-bit signed |
| `RaIdx` | NumberToken | Accumulator indices |
| `RbIdx` | NumberToken | Cyclic register indices |

---

## New Query Functions

### Get Instruction by Name
```python
from ipu_common.instruction_spec import get_instruction

inst = get_instruction("xmem", "ldr_mult_reg")
# Returns: {"operands": [...], "doc": ..., "execute_fn": "..."}
```

### Get Instruction by Opcode (Position)
```python
from ipu_common.instruction_spec import get_instruction_by_opcode

name, inst = get_instruction_by_opcode("lr", 1)  # opcode 1
# Returns: ("set", {"operands": [...], ...})
```

### Get Operand Names and Types
```python
from ipu_common.instruction_spec import get_operand_names_and_types

operands = get_operand_names_and_types("xmem", "ldr_mult_reg")
# Returns: [("dest", "MultStageReg"), ("offset", "LrIdx"), ("base", "CrIdx")]
```

### Extract All Opcodes
```python
from ipu_common.instruction_spec import extract_opcodes

opcodes = extract_opcodes()
# Returns:
# {
#     "xmem": ["str_acc_reg", "ldr_mult_reg", ...],
#     "lr": ["incr", "set", "add", "sub"],
#     ...
# }
```

---

## Test Results

### Assembler Tests ✅
```bash
cd src/tools/ipu-as-py && python3 -m pytest test/test_assemble.py -v
# PASSED
```

### Emulator Tests ✅
```bash
cd src/tools/ipu-emu-py && python3 -m pytest test/test_regfile.py -v
# 34 passed
```

---

## Adding a New Instruction

**Workflow** (now vastly simplified):

1. **Edit** `src/tools/ipu-common/src/ipu_common/instruction_spec.py`:
   ```python
   "my_new_slot": {
       "existing_inst": { ... },
       # ... existing instructions ...
       "new_instruction": {
           "operands": [
               {"name": "src", "type": "LrIdx"},
               {"name": "dst", "type": "CrIdx"},
           ],
           "doc": InstructionDoc(
               title="New Instruction",
               summary="Does something",
               syntax="new_instruction src dst",
               operands=["src: Source", "dst: Destination"],
               operation="dst = src",
           ),
           "execute_fn": "execute_new_instruction",
       },
   }
   ```

2. **Automatic**:
   - Assembler sees `MySlotOpcode.enum_array()` includes `"new_instruction"`
   - Emulator sees `MY_SLOT_OP_NEW_INSTRUCTION` constant
   - Position derived automatically

3. **No manual editing** of:
   - `opcodes.py` class definitions
   - `execute.py` constants
   - Opcode assignments
   - Anything!

---

## Validation

**Automatic validation** runs on module import:

- All slots have instructions
- All instructions have required fields (operands, doc, execute_fn)
- All operands have name and type
- All operand types are recognized
- Structure is correct

Raises `ValueError` if validation fails.

**Test it**:
```python
from ipu_common.instruction_spec import validate_instruction_spec
validate_instruction_spec()
# Runs automatically on import; OK = silent, ERROR = ValueError
```

---

## Files Modified

| File | Change | Reason |
|------|--------|--------|
| `src/tools/ipu-common/src/ipu_common/instruction_spec.py` | Complete refactor | Position-based, structured operands, runtime factories |
| `src/tools/ipu-as-py/src/ipu_as/opcodes.py` | Import from factory | Single source of truth |
| `src/tools/ipu-emu-py/src/ipu_emu/execute.py` | Import from factory | Single source of truth |
| `src/tools/ipu-common/src/ipu_common/codegen.py` | **DELETED** | No longer needed |

---

## Before/After Comparison

### Before: Opcode Duplication Map
```
instruction_spec.py (opcodes hardcoded)
    ↓
    ├→ codegen.py (code generation pipeline)
    │   ↓ (generates)
    │   opcodes_generated.py
    │       ├→ ipu-as-py/opcodes.py (manual import)
    │       └→ ipu-emu-py/execute.py (manual update)
```
**Problems**: Duplication, sync issues, manual updates required

### After: Single Source of Truth
```
INSTRUCTION_SPEC (instruction_spec.py)
    ├→ create_assembler_opcodes() factory
    │   ↓
    │   XmemInstOpcode, LrInstOpcode, ...
    │   (ipu-as-py uses these)
    │
    └→ create_emulator_constants() factory
        ↓
        XMEM_OP_*, LR_OP_*, ... constants
        (ipu-emu-py uses these)
```
**Benefits**: No duplication, automatic sync, runtime generation

---

## Summary

✅ **Problem 1 (Explicit Opcodes)** → Position-based auto-derivation
✅ **Problem 2 (Code Generation)** → Runtime factories, no file writing
✅ **Problem 3 (Unstructured Operands)** → Named, typed operands

**Key Achievement**: Instructions are now the *true* source of truth. Add an instruction once, both assembler and emulator see it automatically. No manual opcode management, no code generation, no duplication.

---

## Next Steps (Optional)

- [ ] Use runtime operand names for better error messages
- [ ] Generate documentation automatically from specs
- [ ] Create instruction browser UI using the structured data
- [ ] Generate hardware emulator from INSTRUCTION_SPEC
