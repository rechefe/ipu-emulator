# IPU Emulator

The IPU is a 128-lane VLIW vector processor for neural-network inference. This repository defines its instruction set, assembles programs for it, emulates it, and holds the kernels that run on it.

## Program and encoding

**Instruction word**:
One VLIW word fetched and executed per cycle. It holds one instruction per slot.
_Avoid_: VLIW word, compound instruction, bundle, line

**Slot**:
A fixed position in the instruction word that holds one instruction for one stage (COND, LR0–LR2, LOAD, MULT, ACC, AAQ, STORE, and the emulator-only ACC_STORE and BREAK).
_Avoid_: lane (for slots), slot type, slot name

**Instruction**:
One operation in one slot of an instruction word, written as a mnemonic and operands.
_Avoid_: inst, instr, op (in prose)

**Mnemonic**:
The upper-case name of an instruction. Control-stage instructions use plain names (`ADD`, `BEQ`); datapath instructions use `UNIT.OP[.VARIANT]` (`LD.MULT`, `ACC.ADD.FIRST`). Mnemonics never contain underscores.

**Pseudo-instruction**:
A mnemonic the assembler expands into real instructions (`B`, `BZ`, `BNZ`, `BGT.S`, `BLE.S`). It has no opcode.

**Hardware word**:
The part of the instruction word that real hardware decodes: every slot tagged hardware.
_Avoid_: RTL word

**Emulator word**:
The full instruction word the emulator decodes: the hardware word followed by the emulator-only slots.

**Hardware / emulator-only / assembler-only**:
The tag every slot, instruction and element format carries in the ISA package. Emulator-only parts exist only in the emulator; assembler-only parts are pseudo-instructions.
_Avoid_: simulation-only, debug-only

**pc**:
The index of the instruction word being executed.
_Avoid_: program counter, PC (in code)

## Machine

**Stage**:
A pipeline unit that executes one slot kind: Control (LR and COND), LOAD, MULT, ACC, AAQ, STORE.

**Control stage**:
The stage that executes LR and COND instructions: loop-register arithmetic and branches.

**Lane**:
One of the 128 parallel element positions of a vector.

**Element**:
The value one lane holds. Its width depends on the element format.

**Element format**:
The numeric encoding of elements for a run: INT8 and E1–E7 (8-bit floats with that many exponent bits) on hardware, and the emulator-only FP32 and INT32. It fixes element width, row width and accumulator type.
_Avoid_: dtype, wide mode, wide-vector debug mode, narrow mode

**Accumulator type**:
INT32 for the INT8 format, FP32 for every float format. MULT results and R_ACC use it.

**XMEM**:
The IPU's external memory, addressed in rows.
_Avoid_: DRAM, main memory

**Row**:
The XMEM unit one vector occupies: 128 elements, so 128 bytes in an 8-bit format and 512 bytes in FP32/INT32.
_Avoid_: word, chunk, line (for XMEM units)

**LR (loop register)**:
One of LR0–LR15, the 32-bit scalar registers that Control-stage instructions compute with: loop counters, row indices, addresses.

**CR (configuration register)**:
One of CR0–CR15, scalar registers the host sets before a run and kernels only read. CR0 is 0 and CR1 is 1.

**dstructure**:
The `valid_elements` and `partition` fields held in a CR (by convention CR15) that shape vector operations.

**Cycle start**:
Register values as they were when the current instruction word began. The Control stage reads LRs at cycle start; MULT reads R0, R1, R_CYCLIC and R_MASK at cycle start.
_Avoid_: snapshot (in prose), pre-instruction value

**Forwarded value**:
A value written earlier in the same cycle and seen by a later stage in that cycle: post-LR-write LRs outside the Control stage, MULT_RES in ACC, R_ACC in AAQ.
_Avoid_: live value

## Faults and events

**Fault**:
An illegal state the IPU detects, numbered F-1 to F-9 in the requirements. An enabled fault halts and records its code and pc; a disabled one sets a sticky flag.
_Avoid_: exception, trap, illegal program error

**Arithmetic event**:
Accumulator overflow, a NaN result or quantization saturation. It never halts; it sets a sticky flag.

**Configuration error**:
A mistake detected before the first cycle, such as an unsupported element format or invalid kernel parameters. It is not a fault.

## Kernels

**Kernel**:
An assembly program implementing one operation, plus its declaration in `kernel.toml`.
_Avoid_: app, application, harness (for the kernel itself)

**Family**:
A group of kernels for one kind of operation (projections, convolutions, softmax), often sharing one template.

**Variant**:
A kernel defined as an entry in a family's registry over a shared template, differing only in parameters such as shape or activation.

**Region**:
A named block of rows in XMEM that holds one kernel input or output.
_Avoid_: buffer, image, blob

**ABI**:
The mapping from a kernel's named values (region bases, sizes, scalars) to the CRs its program reads.
_Avoid_: CR map, calling convention

**Case**:
One concrete test run of a kernel: parameter values, inputs (generated or user-supplied) and how the output is judged.
_Avoid_: test vector, scenario

**Reference**:
The independent computation of a kernel's expected output, normally in PyTorch, rounded to the element format.
_Avoid_: golden model, oracle

**Golden**:
The expected output a case is compared against: the reference result or a user-supplied file.

**Tolerance tier**:
A named, shared accuracy bound for comparing a kernel output with its golden in a given element format.

**Harness**:
The framework that runs a kernel case: places regions, packs data, programs CRs, assembles, runs and compares.

## Emulator as reference model

**Reference model**:
The emulator used through its stable API as the expected behaviour for RTL verification.

**Commit record**:
What one instruction word did: pc before and after, the instruction retired in each slot, register and XMEM writes, and any halt or fault.

**Trace**:
The sequence of commit records of one run.

**Prototyping mode**:
An opt-in emulator mode that skips work proven not to change results, for faster runs.
