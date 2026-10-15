# Fault model: F-1 to F-9 with IDs and a halt record

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-19 · Size M · Depends on: E04-03, E04-07, E03 (fault list F-1..F-9 in `ipu_isa`) · Behaviour change: yes (every illegal-program error becomes an `EmulatorFault` with a fault ID; the state records a halt with fault code and PC)

## Goal

The emulator reports illegal programs as the hardware faults in `docs/content/requirements/faults.md` (Q52, Q59). Each fault has its ID (F-1 to F-9). On a fault, the state enters a halted status and records the fault code and the PC of the faulting word (IPU-FLT-4). There is no separate `SimulationLimit` class: the cycle limit is F-9 (E04-21), and F-8 is listed as not modelled.

This issue covers IDs, classification and the halt record. Precision (no partial state) is E04-20; enable bits are E04-22.

## Background

Today's illegal-program errors, mapped to the fault list:

| Fault | Illegal state (faults.md) | Today |
|---|---|---|
| F-1 | reserved opcode or operand encoding | undefined opcode: `IndexError` from the plan tuples (`_build_slot_plans`, ipu.py:502); mult-stage operand not R0/R1: `EmulatorError` (ipu.py:823); LD.MULT `dest` (ipu.py:1055); INC.MODP2 `k` (ipu.py:1163) |
| F-2 | fetch at or beyond the program length | not detected; `PC >= INST_MEM_SIZE` is the normal halt (`IpuState.is_halted`, ipu_state.py:247) |
| F-3 | two LR lanes write the same LR | `RuntimeError` (ipu.py:1203, 2244) |
| F-4 | LD.MULT.CYCLIC index not 0/128/256/384 | `EmulatorError` (ipu.py:1083) |
| F-5 | ACC.RESHAPE mask above 8, or index of 128 or more | `EmulatorError` (ipu.py:1598, 1607) |
| F-6 | load/store row beyond the table | XMEM row checks: `EmulatorError` (ipu.py:666-680); `XMem` bounds: `ValueError` (xmem.py:60, 73) |
| F-7 | access to a read-only/write-only table | not modelled (the emulator has no table configuration) |
| F-8 | stall above a limit | not modelled (Q59: no stall model) |
| F-9 | cycle limit | `RuntimeError` in `run_until_complete`/`run_with_debug` (emulator.py:76, 122); E04-21 |

All 89 `.asm` files under `src/tools/ipu-apps/src/ipu_apps` end with BKPT (HALT after E03's rename), so F-2 does not affect kernels. Emulator unit tests that run off the end do.

## Implementation

1. `EmulatorFault(EmulatorError)` gains `fault: FaultId` (enum from `ipu_isa`, carrying the number and description) and `pc: int`.
2. `IpuState` gains `halt: HaltRecord | None`:
   ```python
   @dataclass(frozen=True)
   class HaltRecord:
       reason: Literal["halt", "fault"]
       pc: int
       fault: FaultId | None
   ```
   `is_halted` becomes `halt is not None`. HALT sets `HaltRecord("halt", pc, None)` after the word retires.
3. Add `IpuState.program_length`, set by `load_program`. Fetch at `pc >= program_length` raises F-2. A branch target beyond it raises F-2 at the next fetch.
4. Replace each raise in the table with `raise EmulatorFault(FaultId.F_n, pc, detail)`. `Cycle` catches it, records `HaltRecord("fault", ...)`, and re-raises.
5. `run_until_complete` keeps raising (now `EmulatorFault`) so existing callers fail loudly; it accepts `raise_on_fault=False` to return instead. The reference API (E04-30) returns the fault in the commit record.
6. Open questions recorded in the PR (do not decide here):
   - `pad_mode` that needs floating-point lanes under an integer format (ipu.py:983): F-1 (reserved operand encoding) or a configuration error?
   - Wide-mode R_CYCLIC alignment (ipu.py:681-686): does it become F-4 under element formats?
   - F-6/F-7 need a table model. Until one exists, F-6 covers "row beyond the XMEM address space"; F-7 is never raised.
   - `ACTIVATE.QUANTIZE requires INT8 mode` (ipu.py:1815) is AaQ: record in E09 open questions.
7. Update the tests that expect `RuntimeError`/`ValueError`/`IndexError` for these cases to expect `EmulatorFault` with the right ID. Add HALT to emulator unit-test programs that rely on running off the end (the E04-01 helper can append it).

## Tests

- [ ] One test per modelled fault (F-1 to F-6, F-9): the right `FaultId`, the faulting PC, and `state.halt.reason == "fault"`.
- [ ] HALT sets `state.halt.reason == "halt"` and the PC of the HALT word.
- [ ] Running past `program_length` raises F-2.
- [ ] E02 kernel output digests unchanged (no kernel faults).
- [ ] `test_prototyping_equivalence.py` under `--define=ipu_proto=1`: both modes raise the same fault at the same PC.

## Acceptance Criteria

- [ ] No `RuntimeError`, `ValueError` or `IndexError` escapes a cycle for an illegal program.
- [ ] F-7 and F-8 are documented as not modelled in the fault table of the emulator docs.
- [ ] The PR description lists every test whose expected exception changed.
