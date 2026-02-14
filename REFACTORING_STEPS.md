# Python Packages Refactoring: Unified IPU Definitions

## Problem Analysis

### Current Duplication

#### 1. **Opcodes Duplication**
- **`ipu-as-py/opcodes.py`**: Defines opcodes as enum lists
  ```python
  class XmemInstOpcode(Opcode):
      @classmethod
      def enum_array(cls):
          return ["str_acc_reg", "ldr_mult_reg", "ldr_cyclic_mult_reg", "ldr_mult_mask_reg", "xmem_nop"]
  ```
- **`ipu-emu-py/execute.py`**: Re-defines as integer constants  
  ```python
  XMEM_OP_STR_ACC_REG = 0
  XMEM_OP_LDR_MULT_REG = 1
  XMEM_OP_LDR_CYCLIC_MULT_REG = 2
  XMEM_OP_LDR_MULT_MASK_REG = 3
  XMEM_OP_XMEM_NOP = 4
  ```
- **Problem**: Manual mapping required; indices must match enum_array order

#### 2. **Registers Duplication**
- **`ipu-as-py/reg.py`**: Defines registers as EnumToken classes
  ```python
  IPU_LR_REG_NUM = 16
  LR_REG_FIELDS = [f"lr{i}" for i in range(IPU_LR_REG_NUM)]
  
  class LrRegField(ipu_token.EnumToken):
      @classmethod
      def enum_array(cls):
          return LR_REG_FIELDS
  ```
- **`ipu-emu-py/descriptors.py`**: Re-defines as RegDescriptor objects
  ```python
  RegDescriptor(name="lr", kind=RegKind.LR, size_bytes=4, count=16, dtype=RegDtype.UINT32)
  ```
- **Problem**: Two different representations; changes in one don't propagate to the other

#### 3. **Instruction Structure Duplication**
- **`ipu-as-py/inst.py`**: Instruction classes for parsing/validation
- **`ipu-emu-py/execute.py`**: Hardcoded field indices and dispatch logic
- **Problem**: Instruction format changes require updates in multiple places

---

## Solution Design

### Target Architecture
```
ipu-common/
├── src/
│   └── ipu_common/
│       ├── __init__.py
│       ├── types.py              # Shared types (RegDtype, RegKind, etc.)
│       ├── registers.py          # Single source of truth for registers
│       ├── instruction_spec.py   # SINGLE SOURCE: All instruction definitions
│       └── codegen.py            # Auto-generate opcodes, enums, constants
└── BUILD.bazel
```

### Inversion of Dependency

**Old approach**: Define opcodes → define instructions  
**New approach**: Define instructions → **auto-generate** opcodes from them

This eliminates the need to write opcodes twice:
```python
# In ipu_common/instruction_spec.py — ONE place to define
INSTRUCTION_SPEC = {
    "xmem": {
        "str_acc_reg": {"opcode": 0, "operands": [...], "execute": handler_fn},
        "ldr_mult_reg": {"opcode": 1, "operands": [...], "execute": handler_fn},
        ...
    },
    "lr": {
        "incr": {"opcode": 0, "operands": [...], "execute": handler_fn},
        ...
    }
}

# codegen.py extracts opcodes automatically
opcodes = extract_opcodes(INSTRUCTION_SPEC)
# → {"xmem": ["str_acc_reg", "ldr_mult_reg", ...], "lr": ["incr", ...]}
```

### Key Changes

#### Phase 1: Extract Shared Foundations
1. Move `ipu_common/registers.py` - unified register definitions
   - Single `REGISTERS` mapping with all metadata
   - Auto-generate EnumToken classes for assembler
   - Auto-generate RegDescriptor for emulator
   
2. Create `ipu_common/instruction_spec.py` - **instructions as source of truth**
   - Each instruction defined once with: opcode, operands, format, execution logic
   - Includes both assembler metadata (parsing) and emulator metadata (execution)

3. Create `ipu_common/codegen.py` - factory functions to extract derived data
   - `extract_opcodes()` → generates opcode enums automatically
   - `create_assembler_opcodes()` → generates Opcode subclasses 
   - `create_emulator_opcode_constants()` → generates `XMEM_OP_*` constants
   - All derived from instruction spec, never manually written

#### Phase 2: Update Package Dependencies
1. `ipu-as-py` imports registers & generated opcodes from `ipu-common`
2. `ipu-emu-py` imports registers & generated opcodes from `ipu-common`
3. Remove local opcode definitions entirely

#### Phase 3: Link Instruction Implementations
1. `ipu-emu-py/execute.py` uses instruction spec for dispatch
2. Execution handlers defined alongside instruction metadata
3. No manual opcode indices needed

---

## Implementation Steps

### Step 1: Create ipu_common Foundation ✅ COMPLETE
- [x] Add `ipu_common/types.py` with shared enums (RegDtype, RegKind, RegDescriptor)
- [x] Create `ipu_common/__init__.py` to export types
- [x] Create BUILD.bazel and pyproject.toml for ipu-common package
- [x] Verify ipu-common builds and exports these types
- [x] No external changes yet

### Step 2: Unify Register Definitions ✅ COMPLETE
- [x] Create `ipu_common/registers.py` with master `REGISTER_DEFINITIONS` dict
- [x] Add factory function: `create_assembler_reg_classes()` → generates EnumToken classes
- [x] Add factory function: `create_regfile_schema()` → generates RegDescriptor list
- [x] Add tests to verify both paths produce consistent results
- [x] Update `ipu-as-py/reg.py` to import from ipu-common (keep old interface)
- [x] Update `ipu-emu-py/descriptors.py` to import from ipu-common (keep old interface)
- [x] Update BUILD.bazel files to add ipu-common dependency
- [x] Verify all existing tests pass

### Step 3: Create Master Instruction Specification
- [ ] Create `ipu_common/instruction_spec.py` with `INSTRUCTION_SPEC` dict
  ```python
  INSTRUCTION_SPEC = {
      "xmem_slot": {
          "str_acc_reg": {"opcode": 0, "operands": [...], "execute": execute_str_acc_reg},
          "ldr_mult_reg": {"opcode": 1, "operands": [...], "execute": execute_ldr_mult_reg},
          ...
      },
      "lr_slot": {
          "incr": {"opcode": 0, "operands": [...], "execute": execute_lr_incr},
          ...
      },
      ...
  }
  ```
- [ ] Each instruction defined once with all metadata (assembler + emulator)
- [ ] Include operand types, format info, doc strings, execution handlers

### Step 4: Auto-generate Opcodes from Instructions
- [ ] Create `ipu_common/codegen.py` with:
  - `extract_opcodes(spec)` → returns `{"xmem": ["str_acc_reg", ...], "lr": [...]}`
  - `create_assembler_opcodes(spec)` → generates Opcode subclasses dynamically
  - `create_emulator_opcode_constants(spec)` → generates `XMEM_OP_STR_ACC_REG = 0`, etc.
- [ ] All functions derived from `INSTRUCTION_SPEC` — **never manual duplication**
- [ ] Add tests: verify opcode indices match instruction definitions

### Step 5: Update ipu-as-py
- [ ] Update `ipu-as-py/opcodes.py` to import generated classes from ipu-common
- [ ] Update `ipu-as-py/inst.py` to reference instruction metadata from ipu-common
- [ ] Remove manual opcode definitions
- [ ] Run tests: should pass unchanged

### Step 6: Update ipu-emu-py  
- [ ] Update `ipu-emu-py/execute.py` to import opcode constants from ipu-common
- [ ] Update instruction dispatch to use `INSTRUCTION_SPEC` handlers
- [ ] Remove hardcoded opcode constants (`XMEM_OP_*`, etc.)
- [ ] Remove duplicate field indices
- [ ] Run tests: should pass unchanged

### Step 7: Validate Integration
- [ ] Run existing tests for both packages (should pass unchanged)
- [ ] Create integration test: assemble → load into emulator → execute
- [ ] Verify: adding a new instruction updates opcodes automatically
- [ ] Verify: modifying an instruction doesn't break either package

### Step 8: Clean Up & Document
- [ ] Update package README files
- [ ] Add docstrings explaining instruction spec format
- [ ] Create migration guide for developers adding new instructions
- [ ] Archive old definition files (optional deletion)

---

## Success Criteria

✅ Instructions are the single source of truth  
✅ Opcodes are **automatically extracted** from instruction definitions (zero manual duplication)  
✅ Adding a new instruction automatically creates corresponding opcodes  
✅ Single source of truth for registers (no schema duplication)  
✅ All existing tests pass without modification  
✅ Both packages can import unified definitions  
✅ No breaking changes to public APIs  
✅ Execution handlers co-located with instruction metadata  

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking assembler API | Keep old `ipu_as.opcodes` interface; import generated classes from common | 
| Breaking emulator API | Keep old `ipu_emu.execute` constants; import from common |
| Test failures | Comprehensive tests before integrating; validate opcode extraction |
| Circular imports | Careful module design; separate spec from execution handlers if needed |
| Build system issues | Update BUILD.bazel files explicitly; test bazel build |
| Instruction spec complexity | Start with simple format; use clear documentation |

---

## Timeline Estimate

- Phase 1 (types extraction): ~1-2 hours
- Phase 2 (register unification): ~2-3 hours  
- Phase 3 (master instruction spec): ~3-4 hours
- Phase 4 (opcode codegen): ~2-3 hours
- Phase 5-6 (package updates): ~4-5 hours
- Phase 7-8 (validation & cleanup): ~2 hours

**Total: ~14-18 hours**

