# Spike: minimal `IPU` ETISS architecture executing a 224-bit NOP program

Part of the [ETISS integration epic](00-epic.md). **Go/no-go gate** for the epic.

## Goal

Prove, on the real ETISS code base, the three assumptions the design depends
on before any generator or Bazel work starts:

1. ETISS fetches, decodes and JIT-compiles a **224-bit** instruction word
   (`VariableInstructionSet` width 224; `BitArray` is a `boost::dynamic_bitset`,
   but the width is outside the "usual" 16/32/64/128/256 set).
2. A single catch-all `InstructionDefinition` (code = mask = all-zero
   `BitArray`) is a workable way to dispatch a VLIW word; the callback can
   slice slot fields with `BitArrayRange` and emit calls into a helper C
   library that the JIT links (same mechanism as `RV32IMACFDFuncs.c`).
3. Halt-by-address (`instructionPointer >= IMEM_END` → `CPUFINISHED`) and a
   1024-word NOP-padded IMEM image reproduce the Python cycle count.

## Implementation

Throw-away code is acceptable; the point is the answers, not the artefacts.

- Build ETISS from source at a pinned commit (`74451e0`, 2026-08-12) with the
  bundled TCC JIT (`cmake -S etiss -B build -DCMAKE_BUILD_TYPE=Release`;
  needs `libboost-{system,filesystem,program-options}-dev`).
- Create `spike/IPU/` modelled on `ArchImpl/RV32IMACFD`:
  - `IPU.h`: `struct IPU { ETISS_CPU cpu; etiss_uint32 LR[16]; etiss_uint32 CR[16]; etiss_uint8 R[2][128]; etiss_uint64 cycles; }`.
  - `IPUArch.cpp`: `newCPU/resetCPU/deleteCPU`, `getInstructionSizeInBytes() = 28`,
    `getMaximumInstructionSizeInBytes() = 28`, `initInstrSet` creating
    `ModedInstructionSet → VariableInstructionSet(224) → InstructionSet(224)`,
    `VirtualStruct` with `LR0..15`, `CR0..15`, `PC` (word index = `instructionPointer / 28`).
  - `IPUInstr.cpp`: one `InstructionDefinition` whose callback reads the
    **cond-slot** opcode and label field at the bit offsets printed by
    `CompoundInst.get_fields()` and emits either `nextPc = addr + 28` or
    `nextPc = label * 28` (BEQ with CR0,CR0 = unconditional), plus
    `ipu_count_cycle(cpu)` from `IPUFuncs.c`, and the halt check.
  - `IPUArchLib.cpp`: the `ETISS_LIBNAME` boilerplate.
  - `CMakeLists.txt`: `FIND_PACKAGE(ETISS)` + `ETISSPluginArch(IPU)`.
- Runner: either `bare_etiss_processor` with an ini (`arch.cpu=IPU`,
  `jit.type=TCCJIT`, `simple_mem_system.memseg_*` for IMEM at `0x0`), or a
  20-line `main.cpp` that calls `etiss::loadLibrary()` on `libIPU.so`.
- Test program (assembled with `ipu-as --format bin`):

  ```asm
  start:  NOP;;
          NOP;;
          B start_end;;
  filler: NOP;;
  start_end: NOP;;
  ```

  padded to 1024 words with the canonical NOP word; expect the run to halt
  with `CPUFINISHED` after exactly the cycle count `run_until_complete()`
  reports for the same binary.

## Questions to answer (write the answers into the spec §3 / §12)

- [ ] Does translation of a 224-bit `InstructionSet` work with TCC and GCC JITs?
      If not, does width 256 with a 32-byte stride work (fallback)?
- [ ] Does `etiss.max_block_size` / block caching behave with 28-byte
      instructions (blocks end on branches as expected)?
- [ ] How are helper symbols from `libIPU.so` resolved by the JIT (global
      symbol visibility vs `jit.external_libs`)? Record the required config.
- [ ] Is a custom return code (e.g. `IPU_BREAK`) definable from the arch, or
      must `BREAKPOINT` (-19) be reused?
- [ ] Wall-clock for a 1M-cycle loop on TCC vs the Python emulator (rough
      order of magnitude only).

## Acceptance Criteria

- [ ] The NOP/branch program above executes on ETISS and halts with the same
      cycle count as the Python emulator.
- [ ] The five questions are answered in the spec; the fallback (256-bit
      width) is either not needed or adopted.
- [ ] Decision recorded: go / no-go for issues 2–6.
