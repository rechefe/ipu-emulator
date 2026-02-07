# IPU Python Emulator — Refactoring Plan

## Motivation

Rewrite the C emulator in Python to:

1. **Tight integration with the Python assembler** — the assembler already defines the ISA (opcodes, bitfields, encoding). The emulator should reuse `ipu_as` instruction classes directly, eliminating the generated `inst_parser.h` intermediary for the emulator path.

2. **Descriptor-driven architecture** — registers, debug commands, and state serialization are all generated from declarative descriptors. Adding a new register or debug view means adding one descriptor entry, not touching 3+ files.

3. **Assembler ↔ emulator linkage** — each `Inst` subclass (e.g. `MultInst`, `AccInst`) owns *both* encode/decode *and* execute logic, making the ISA self-documenting and impossible to get out of sync.

## Language Choice: Python

- The emulator is cycle-functional, not performance-critical at MHz scale.
- Developer velocity and integration with the existing `ipu_as` package are the primary wins.
- NumPy can accelerate the hot path (128-byte vector dot products) if needed later.

## Architecture

New package: `src/tools/ipu-emu-py/` (sibling to `ipu-as-py`), importable as `ipu_emu`.
Depends on `ipu_as` for instruction definitions, opcodes, and register field enums.

```
src/tools/ipu-emu-py/
├── pyproject.toml
├── pytest.ini
├── src/
│   └── ipu_emu/
│       ├── __init__.py
│       ├── descriptors.py      # Register descriptors (declarative schema)
│       ├── xmem.py             # External memory model (2 MB flat)
│       ├── regfile.py           # Register file (built from descriptors)
│       ├── ipu_state.py         # IPU top-level state (regfile + xmem + PC + inst_mem)
│       ├── execute.py           # VLIW dispatch loop + per-slot executors
│       ├── debug_cli.py         # Interactive debug CLI (auto-generated from descriptors)
│       └── emulator.py          # High-level runner (run_until_complete, run_test, etc.)
└── test/
    ├── test_xmem.py
    ├── test_regfile.py
    ├── test_ipu_state.py
    └── test_execute.py
```

## Phased Implementation

### Phase 1 — Foundation: XMEM + RegFile + Descriptors ✅

- [x] `descriptors.py` — `RegDescriptor` dataclass defining name, size, count, dtype, debug formatter
- [x] `xmem.py` — 2 MB `bytearray`, with `read/write_address`, `load_array_to`, `load_matrix_to`
- [x] `regfile.py` — `RegFile` class built from `REGFILE_SCHEMA` (list of `RegDescriptor`s)
  - R registers (2 × 128 B), R cyclic (512 B with wraparound), R mask (128 B)
  - Accumulator (512 B, byte + word views), LR (16 × u32), CR (16 × u32)
  - Snapshot support (deep copy for VLIW read-before-write semantics)
- [x] `ipu_state.py` — `IpuState` combining regfile, xmem, PC, misc forwarding regs
- [x] Unit tests for all of the above (48 tests)

### Phase 2 — VLIW Dispatch + Instruction Executors ✅

- [x] `ipu_math.py` — INT8, FP8_E4M3, FP8_E5M2 multiply/add using `ml_dtypes`
- [x] `execute.py` — `execute_next_instruction(state)` with VLIW snapshot semantics
  - Instruction decoder from `CompoundInst.get_fields()` field layout
  - Break, XMEM, LR (×2 slots, conflict detection), Mult (ee/ev/ve + mask+shift), Acc, Cond executors
- [x] `emulator.py` — `run_until_complete`, `run_with_debug` with breakpoint callback
- [x] Tests: 31 tests covering registers, memory, control flow, breakpoints, FP8, decode round-trip

### Phase 3 — Debug CLI (auto-generated from descriptors) ✅

- [x] `debug_cli.py` — GDB-like CLI: `regs`, `get <reg>`, `set <reg> <val>`, `step`, `continue`, `disasm`, `save`
- [x] All register read/display commands auto-generated from `REGFILE_SCHEMA`
- [x] JSON state export (structure matches C debug output for comparison)
- [x] Disassembly via `CompoundInst.decode()` round-trip
- [x] Tests: 44 tests covering register resolution, all commands, debug levels, JSON export, formatting

### Phase 4 — High-level Emulator + App Migration ✅

- [x] `emulator.py` — `run_test` orchestrator, `load_binary_to_xmem`, `dump_xmem_to_binary`, `load_fp32_as_fp8_to_xmem`
- [x] `ipu_math.py` — `fp32_to_fp8_bytes`, `fp8_bytes_to_fp32` FP32↔FP8 conversion via `ml_dtypes`
- [x] **Bug fix**: `load_binary_instructions` now uses 32-bit word-aligned instruction size (matching assembler output)
- [x] `apps/fully_connected.py` — Full port of C harness: setup (load inputs, transpose weights, set CRs), teardown (dump outputs), `run_fully_connected` entry point
- [x] End-to-end tests: assemble → load → run → validate against golden outputs (INT8 + FP8_E4M3)
- [x] Tests: 27 tests covering binary I/O, FP32→FP8 conversion, weight transpose, FC setup/teardown, run_test orchestration, end-to-end golden validation

### Phase 5 — Convergence + C Deprecation ✅

- [x] Run both C and Python emulators on all test programs, diff outputs
  - `test_parity.py` — 21 tests providing 1:1 mapping of every C++ GTest
  - `test_emulator_e2e.py` — INT8, FP8_E4M3, FP8_E5M2 end-to-end golden validation
  - Python test suite exceeds C++: adds beq/blt/bnz/bz, breakpoints, decode round-trip, FP8 mult
- [x] Once parity is proven, mark C emulator as deprecated
  - Deprecation notices added to `emulator.h`, `emulator.c`, `fully_connected.c`, `ipu_test_helper.h`
  - Bazel `deprecated` tags on `emulator`, `ipu_test_helper`, and `ipu_emulator_tests` targets
- [x] Update Bazel build to use Python emulator as the default path
  - `test_parity` target added to Python emulator BUILD
  - All 6 Python test targets pass: test_xmem, test_regfile, test_execute, test_debug_cli, test_parity, test_emulator_e2e
- [x] Update documentation
- [x] **Total: 172 tests passing** (150 + 21 parity + 1 FP8_E5M2 E2E)

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python | Integration with `ipu_as`, developer velocity |
| Memory model | `bytearray` | Simple, efficient for byte-level access |
| Register storage | `bytearray` + `numpy` views | Zero-copy word views via `np.frombuffer` |
| Instruction dispatch | Method on `Inst` subclass | Single source of truth for ISA |
| Debug CLI | Auto-generated from descriptors | Zero maintenance cost for new registers |
| Test strategy | Port C++ tests to pytest, keep C tests as regression | Ensures parity during migration |