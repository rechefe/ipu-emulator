# IPU Emulator

The IPU is a 128-lane VLIW vector processor for neural-network inference. This repository defines its instruction set, assembles programs for it, emulates it, and holds the kernels that run on it.

## Program and encoding

**Instruction word**:
One VLIW word fetched and executed per cycle. It holds one instruction per slot.
_Avoid_: VLIW word, compound instruction, bundle, line

**Slot**:
A fixed position in the instruction word that holds one instruction for one stage (COND, LR0–LR2, LOAD, MULT, ACC, AAQ, and the emulator-only AAQ_EXACT and BREAK).
_Avoid_: lane (for slots), slot type, slot name

**Instruction**:
One operation in one slot of an instruction word, written as a mnemonic and operands.
_Avoid_: inst, instr, op (in prose)

**Mnemonic**:
The upper-case name of an instruction. Control-stage instructions use plain names (`ADD`, `BEQ`); datapath instructions use `UNIT.OP[.VARIANT]` (`LD.MULT`, `ACC.ADD.FIRST`). Mnemonics never contain underscores.

**Pseudo-instruction**:
A mnemonic the assembler expands into real instructions (`B`, `BZ`, `BNZ`, `BGT.S`, `BLE.S`). It has no opcode.

**`HALT` instruction**:
The COND instruction that ends a kernel: the kernel is done when the word that holds it retires. Not the host's `HALT` command, which stops a running kernel in the `HALTED` state.

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
A pipeline unit that executes one slot kind: Control (LR and COND), LOAD, MULT, ACC, AAQ. The AaQ stage activates, quantizes and stores its own row (ADR 0007); there is no STORE stage.

**Control stage**:
The stage that executes LR and COND instructions: loop-register arithmetic and branches.

**Lane**:
One of the 128 parallel element positions of a vector.

**Element**:
The value one lane holds. Its width depends on the element format.

**Element format**:
The numeric encoding of elements for a run, one per mode (ADR 0007): NX8 on hardware, and the emulator-only FP32. It fixes element width, row width and whether rows carry metadata. INT8, INT32 and the former biased E1–E7 formats are gone.
_Avoid_: dtype, wide mode, wide-vector debug mode, narrow mode

**Mode**:
Which element format a run is in: NX8 (the hardware) or FP32 (emulator-only, for debugging a kernel that will run in NX8). Chosen when the state is built; every load and store moves one row of the mode.

**NX8**:
The hardware's element format: a row of 128 elements of at most 8 bits plus a metadata word. Each element's exponent is a distance below the row's scale; its type (sign, exponent width fe, width W, dynamic) comes from a QF half.

**NX8 type**:
One QF value, such as e4 (signed, fe 4, 8 bits) or dynamic (fe 1 or 2 per row). A kernel names its NX8 types in `kernel.toml` `formats` (`e1`–`e7`, `dynamic`, `s<sign>e<fe>w<bits>`), beside `fp32`.

**Metadata word**:
The 16-bit word each NX8 row carries: `scale[7:0]` (an `e8m0` exponent, bias 127) and `dynamic_exponent[8]`. XMEM keeps one per 128-byte unit; `LD.MULT` copies it into R0, R1 or an R_CYCLIC slot, and `ACTIVATE.QUANTIZE` writes it.

**QF (quantization format)**:
A 16-bit CR half naming an NX8 type: `num_bits[3:0]`, `exp_format[7:4]`, `sign[11:8]`, `dynamic[15:12]`. MULT decodes its vectors with one, `ACTIVATE.QUANTIZE` quantizes to one. The FP32 mode ignores it.

**DS (data structure)**:
A 16-bit CR half shaping a vector operation: `valid_elements[7:0]`, `num_partitions[9:8]` (1, 2, 4 or 8), `mask_type[11:10]`, `valid_partitions[15:12]`. By convention CR15.L. It replaces the old 32-bit dstructure.

**CR half**:
The low or high 16 bits of a CR, written `CR13.L` or `CR13.H`. DS and QF operands always name one; an LR or COND source may read one, sign-extended.

**MULT constant**:
A CR read as a scalar by `MULT.VE`, `MULT.EE` or `MULT.RC.VE`: in NX8 one element with its own scale and type, in FP32 a float32. CR0 and CR1 read as 0.0 and 1.0 in both modes.

**Accumulator type**:
FP32 in both modes: MULT expands each NX8 element to FP32, and MULT results and R_ACC hold FP32.

**Element codec**:
The encoding of one NX8 element and back: decode a code with its row's scale, quantize a row with round-to-nearest (ties up) and saturation. It lives in `ipu_isa.nx8`.

**Lane codec**:
The emulator's numerics for one mode (`ipu_emu.numerics`): decodes a row of lanes (in NX8 with the QF and the row's metadata), multiplies and accumulates. Handlers are written against it.

**XMEM**:
The IPU's external memory, addressed in rows.
_Avoid_: DRAM, main memory

**Row**:
The XMEM unit one vector occupies: 128 elements, so 128 bytes and a metadata word in NX8 and 512 bytes in FP32.
_Avoid_: word, chunk, line (for XMEM units)

**LR (loop register)**:
One of LR0–LR15, the 32-bit scalar registers that Control-stage instructions compute with: loop counters, row indices, addresses.

**CR (configuration register)**:
One of CR0–CR15, 32-bit scalar registers the host sets before a run and kernels only read, each also two CR halves. CR0 is 0 and CR1 is 1.

**dstructure**:
The architecture specs' name for what the ISA calls the DS: see **DS**.

**Cycle start**:
Register values as they were when the current instruction word began. The Control stage reads LRs at cycle start; MULT reads R0, R1, R_CYCLIC, their metadata and R_MASK at cycle start.
_Avoid_: snapshot (in prose), pre-instruction value

**Forwarded value**:
A value written earlier in the same cycle and seen by a later stage in that cycle: post-LR-write LRs outside the Control stage, MULT_RES in ACC, R_ACC in AAQ and AAQ_EXACT.
_Avoid_: live value

**Timing contract**:
The rules for which value, cycle start or forwarded, each register read sees within one cycle, with the slot execution order (ADR 0002). Declared in `ipu_isa.timing`; each instruction declares its register reads and writes.

**Known deviation**:
A read where today's emulator differs from the timing contract, listed in `KNOWN_DEVIATIONS` with the E04 issue that removes it. The list can only shrink. It is empty today.

## Faults and events

**Fault**:
An illegal state the IPU detects, numbered F-1 to F-9 in the requirements. An enabled fault halts and records its code and pc; a disabled one sets a sticky flag.
_Avoid_: exception, trap, illegal program error

**Arithmetic event**:
Accumulator overflow, a NaN result or quantization saturation. It never halts; it sets a sticky flag.

**Configuration error**:
A mistake detected before the first cycle, such as an unsupported element format or invalid kernel parameters, or in a word's check phase, such as an invalid QF or DS half or `ACTIVATE.EXACT` in NX8 (ADR 0007). It is not a fault.

## Kernels

**Kernel**:
An assembly program implementing one operation, plus its declaration in `kernel.toml`.
_Avoid_: app, application, harness (for the kernel itself)

**Family**:
A group of kernels for one kind of operation (projections, convolutions, softmax), often sharing one template. One family, `mobilevit`, has no kernels: it holds the chain tests that run several families' kernels together.

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
