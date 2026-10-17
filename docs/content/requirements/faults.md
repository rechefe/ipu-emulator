# Faults

A fault is an illegal state that the IPU detects. The fault policy is agreed. The fault list is a proposal: [OQ-1](open-questions.md).

## Policy

| ID | Requirement |
|---|---|
| IPU-FLT-1 | The IPU shall detect every illegal state in the fault list. It shall never continue silently with wrong data. |
| IPU-FLT-2 | The IPU shall detect every fault in CTRL or in the cache agent, before the cache agent accepts the faulting word. |
| IPU-FLT-3 | Faults shall be precise. The faulting word and all younger words have no architectural effect. All older words retire. |
| IPU-FLT-4 | On a fault, the IPU shall enter `HALTED` and record the fault code and the PC of the faulting word. |
| IPU-FLT-5 | Each fault shall have an enable bit. When the bit is set, the fault halts. When it is clear, the fault only sets a sticky flag. |
| IPU-FLT-6 | All fault enable bits shall be set after reset. |
| IPU-FLT-7 | After a fault halt, the IPU shall reject `RESUME`, `STEP` and PC writes with an AXI error. The host stops the kernel with `ABORT`. |
| IPU-FLT-8 | When one word raises several faults, the IPU shall record the fault with the lowest number. |
| IPU-FLT-9 | An illegal host access shall not halt the IPU. It returns an AXI error and sets the `HOST_ERR` interrupt status bit. |

IPU-FLT-2 is possible because CTRL resolves every CR and LR operand value before it dispatches a word. Every check in the fault list uses values that CTRL or the cache agent holds.

The behaviour of a word whose fault enable bit is clear is open: [OQ-2](open-questions.md).

## Fault list

Status: **Proposed**. Owner: Yuval Harary.

| ID | Illegal state | Detected in | Proposed behaviour |
|---|---|---|---|
| F-1 | A slot holds a reserved opcode or a reserved operand encoding. | CTRL | Halt |
| F-2 | CTRL fetches at a PC at or beyond the program length, by sequential flow, branch or `BR`. | CTRL | Halt. Needs a program-length register per IMEM bank. |
| F-3 | Two LR lanes of one word write the same LR. | CTRL | Halt |
| F-4 | The index of `LD.MULT.CYCLIC` is not 0, 128, 256 or 384. | CTRL | Halt |
| F-5 | `ACC.RESHAPE` has a mask above 8, or a participating source or destination index of 128 or more. | CTRL | Halt |
| F-6 | A load or a store addresses a table that is not configured, or a row beyond the table size. | Cache agent | Halt |
| F-7 | A load addresses a write-only table, or a store addresses a read-only table. | Cache agent | Halt |
| F-8 | A word stalls for longer than a programmable limit. | Cache agent | Halt. A limit of 0 disables the check. |
| F-9 | A kernel runs for more cycles than a programmable limit. | CTRL | Halt. A limit of 0 disables the check. |

Reserved encodings for F-1, from the instruction reference:

| Slot | Reserved encodings |
|---|---|
| `COND` | One opcode of the 3-bit field |
| `LR` | Seven opcodes of the 4-bit field. `k` outside 1 to 9 in `INC.MODP2`. |
| `LOAD` | `MultStageReg` values other than `R0` and `R1` |
| `MULT` | Two opcodes of the 3-bit field. `MultStageReg` values other than `R0` and `R1`. |
| `ACC` | Three opcodes of the 4-bit field. Values above 2 of `elements_in_row`, `horizontal_stride` and `vertical_stride`. |
| `AAQ` | Activation function codes with no named function. The instruction reference treats them as `identity`: [OQ-6](open-questions.md). |

The fault list and this table are mirrored as data in `ipu_isa.faults`, which derives the reserved encodings from the ISA. A test (`//src/tools/ipu-isa:test_faults_docs`) checks that both tables state the same as that data.

## In the emulator

The emulator raises `ipu_emu.errors.EmulatorFault` for a fault. The exception carries the fault (`FaultId`) and the PC of the faulting word. The cycle that raises it first records `IpuState.halt` as `HaltRecord("fault", pc, fault)`. A `HALT` word records `HaltRecord("halt", pc, None)`. `run_until_complete` re-raises the fault, unless it is called with `raise_on_fault=False`. Faults are precise (IPU-FLT-3): a faulting word changes no register, XMEM byte, PC or statistic, only the halt record and the sticky flag. F-1 and F-3 are found before any slot runs; F-4 to F-6 by a check phase after the LR slot (whose results the other slots see forwarded), which undoes the LR writes on a fault. When one word raises several faults, the lowest number is recorded (IPU-FLT-8).

`IpuState.faults` (`ipu_emu.fault_control.FaultControl`) holds each fault's enable bit, all set at construction, and the sticky flags, which every fault sets, enabled or not (IPU-FLT-5, IPU-FLT-6). With F-1 disabled, the slot with the reserved encoding runs as `NOP` and the rest of the word retires (the proposal in OQ-2). With F-9 disabled, the run goes on past its limit. Clearing the bit of F-2 to F-6 raises `ConfigurationError`, because OQ-2 does not say what the word then does; clearing F-7 or F-8 changes nothing, as neither is modelled.

| Fault | In the emulator |
|---|---|
| F-1 | At fetch: an opcode the ISA does not define, or an operand code that is reserved (the table above) or past its operand type's width (a union column can be wider). |
| F-2 | A fetch at or past `IpuState.program_length`, which `load_program` sets, by sequential flow, branch or `BR`, including a target past IMEM. A state whose IMEM is written directly keeps all of IMEM. |
| F-3 | Two LR sub-slots write the same LR, detected before any LR write. |
| F-4 | An `LD.MULT.CYCLIC` index that is not a slot boundary. |
| F-5 | An `ACC.RESHAPE` mask above 8, or an index of 128 or more. |
| F-6 | An XMEM row beyond the address space, or an access past the XMEM allocation. There is no table model yet, so the row is checked against XMEM. |
| F-7 | Not modelled. The emulator has no table configuration, so no table is read-only or write-only. |
| F-8 | Not modelled. The emulator has no stall model (Q59, [OQ-28](open-questions.md#emulator-and-isa)). |
| F-9 | A run reaches `IpuState.cycle_limit` (default 1 000 000; a runner's `max_cycles` replaces it for that run; 0 disables the check). The fault's PC is the word that would have run next. |

## Arithmetic events

Status: **Proposed**. Owner: Yuval Harary.

Accumulator overflow, NaN results and quantization saturation occur in ACC and AAQ, after the cache agent accepts the word. They are not faults and never halt. Step 0 does not detect them. The fault control registers reserve one sticky flag for each event.

## Illegal host accesses

| Access | Response |
|---|---|
| Write to the IMEM bank, the CR bank or the program-length register of a locked context | AXI error, `HOST_ERR` |
| Write to `CR0` or `CR1` | AXI error, `HOST_ERR` |
| Write to a read-only register | AXI error, `HOST_ERR` |
| Read of the IMEM bank of the active context in a running state | AXI error, `HOST_ERR` |
| Write to the PC outside `HALTED`, or in `HALTED` after a fault | AXI error, `HOST_ERR` |
| Command in a state that does not accept it | AXI error, `HOST_ERR` |
| `ARM` while `tables_valid` of the inactive context is low | AXI error, `HOST_ERR` |
| Write that sets more than one command bit | AXI error, `HOST_ERR` |
| Access to an unmapped address | AXI error, `HOST_ERR` |

## Defined behaviour that is not a fault

The instruction reference defines these cases.

| Case | Behaviour |
|---|---|
| `mask_shift` outside −3 to +3 | Clamps to −3 or +3 |
| `valid_elements` above 128 | Clamps to 128 |
| `dest_slot` of `AGG.*` above 127 | Uses the value modulo 128 |
| `offset` of `ACC.STRIDE` above 3 | Uses the value modulo 4 |
