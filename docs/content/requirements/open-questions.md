# Open questions

An open question blocks the requirement or the block specification that links to it. To close a question, move its answer into the requirement and delete the row.

## Faults

| ID | Question | Proposal | Owner |
|---|---|---|---|
| OQ-1 | Which illegal states does the IPU detect, and what does each do? | The [fault list](faults.md#fault-list), rows F-1 to F-9, and the [arithmetic events](faults.md#arithmetic-events) | Yuval Harary |
| OQ-2 | What does a word do when its fault enable bit is clear? | Decide per fault. For F-1, the slot with the reserved encoding executes as `NOP`. | Yuval Harary |

## Conflicts between existing documents

| ID | Topic | Conflict | Proposal | Owner |
|---|---|---|---|---|
| OQ-3 | CR and LR width | The control spec diagram says 20 bits. Its text and the instruction reference say 32 bits. | 32 bits | Unassigned |
| OQ-4 | IMEM bank depth | The control spec says 128 words per bank and a 7-bit PC. The branch label field is 10 bits. The emulator holds 1024 words. | 1024 words per bank and a 10-bit PC, as a parameter | Unassigned |
| OQ-5 | `SET` source | The instruction reference takes a CR index only. The control spec takes a 5-bit source with a signed immediate mode. | CR index only | Unassigned |
| OQ-6 | AAQ behaviour and element data types | The instruction reference has 13 activation functions in a 4-bit field, requires INT8 mode and clamps to INT8. The AaQ spec has 7 function types in a 3-bit field and quantizes FP32 to a format and a scale. | The AaQ spec is the hardware behaviour. The instruction reference calls its clamp a placeholder. State which element data types the hardware supports. | Unassigned |
| OQ-7 | Store row format | The instruction reference stores 512 bytes. The AaQ spec stores 1039 bits. The cache unit spec has 1024-bit rows with 16 bits of metadata. | 128 elements of 8 bits, plus 16 bits of metadata. The element format is a table property. | Unassigned |
| OQ-8 | Row address to table mapping | CTRL produces `CR[base] + LR[offset]`. The cache unit expects `{table_id, offset[19:0]}`. No document defines the mapping. | The 24-bit sum is `{table_id[3:0], offset[19:0]}`. The base CR carries the table ID. | Unassigned |
| OQ-9 | Load and use in one word | The instruction reference allows `LDR_MULT_REG` and a `MULT` that reads the same register in one word. The control spec tells software to avoid it. | Allowed. The `MULT` slot sees the data the `LOAD` slot of the same word fetched. This matches IPU-ARCH-1. | Unassigned |
| OQ-10 | LR values in the `ACC` slot | The control spec forwards LR values from after the LR writes of the word. The accumulator spec marks `dest_slot`, `source` and `dest` as values from before the word, and `offset` as the value after. | One rule for all slots after CTRL: values from after the LR writes (IPU-ARCH-3) | Unassigned |

## Kernel flow

| ID | Question | Proposal | Owner |
|---|---|---|---|
| OQ-11 | How does a kernel hand a table to the next kernel inside XMEM, with no DRAM round trip? The banks carry the context bit of the first kernel. The second kernel hits only on its own context bit. | The cache unit retags the handed-over banks at `kernel_done`. The table set of the next kernel names the table it inherits. | Cache unit owner |
| OQ-12 | What does the cache unit do with the XMEM banks of a kernel on `kernel_abort`? | The cache unit frees every bank of the aborted context and drains none. | Cache unit owner |
| OQ-13 | How does the host run the kernel in the active context again from `IDLE`? `ARM` always starts the inactive context. | `ARM` takes a context argument in `IDLE`. | Unassigned |

## Interfaces

| ID | Question | Proposal | Owner |
|---|---|---|---|
| OQ-14 | What is the width of the hardware VLIW word, and how does it pack into 32-bit AXI beats? | The emulator word without the `ACC_STORE` and `BREAK` slots, padded to a multiple of 32 bits | Unassigned |
| OQ-15 | What are the AXI address and ID widths, and the width and signal list of `tbl_view`? | Fix in the IPU block specification, together with the cache unit owner | Unassigned |

## Pipeline

| ID | Question | Proposal | Owner |
|---|---|---|---|
| OQ-16 | How do CTRL and the cache agent meet 1 GHz? IPU-ARCH-17 commits LR writes only when the cache agent accepts the word, and IPU-MEM-1 decides in the cycle CTRL presents the word. | Option A: the cache agent decides in the CTRL cycle. Option B: it decides one cycle later and CTRL restores the LR file and the PC when the word is not accepted. Choose in the CTRL and cache agent specifications. | Unassigned |
| OQ-17 | What happens when a word loads a row that an older word in the pipeline stores to? The store reaches XMEM several cycles after the load reads it. Scratch-pad and read-after-write tables allow both accesses. | The cache agent stalls the load until STORE has written the row. | Unassigned |

## Actions on other documents

| ID | Action | Owner |
|---|---|---|
| A-1 | Replace `BKPT` with `END` in the instruction reference, the assembler and the emulator. | Unassigned |
| A-2 | Update the control stage specification. The cache agent issues XMEM reads and stalls CTRL. LR writes commit when the cache agent accepts the word. The host port is AXI4. | Unassigned |
| A-3 | Update the cache unit specification with two table sets, the context bit in tags, `tables_valid`, `ctx_locked`, the bank events and every `CU` expected behaviour. | Cache unit owner |
