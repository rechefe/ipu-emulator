# IPU hardware requirements

The Inference Processing Unit (IPU) is a VLIW vector processor that runs neural-network kernels. Each VLIW word operates on 128-element vectors through a load, multiply, accumulate, activate-and-quantize, and store chain.

The requirements define what the IPU does and what it expects from the blocks around it. The IPU block specification and the sub-block specifications derive from the requirements.

## Specification levels

| Level | Content | Status |
|---|---|---|
| 1. Requirements | Goals, interfaces, main flows, faults, debug | Draft |
| 2. IPU block specification | Ports, registers and timing of the IPU top | Not started |
| 3. Sub-block specifications | CTRL, cache agent, MULT, ACC, AAQ, STORE, host register block | Not started |

## Goals

| ID | Goal |
|---|---|
| IPU-GOAL-1 | Run every kernel in the kernel library: convolution, linear, normalization and shaping, attention, softmax, and MobileViT-S. |
| IPU-GOAL-2 | Execute one VLIW word per clock cycle while no memory access misses. |
| IPU-GOAL-3 | Close timing at 1 GHz on the TSMC 5 nm process. |
| IPU-GOAL-4 | Run consecutive kernels with no host action between them. |
| IPU-GOAL-5 | Detect every illegal state in the fault list. By default, stop with the architectural state intact and report the cause. |
| IPU-GOAL-6 | Give the host halt, single-step, breakpoints and register read-out. Later debug features extend this set without changing halt behaviour. |

The kernel library is the throughput benchmark. Area and power are not constraints at this level.

## Scope

| Item | In scope |
|---|---|
| CTRL, cache agent, MULT, ACC, AAQ and STORE stages | Yes |
| Host register block and debug | Yes |
| Cache unit (CRM, XMEM, DMA) | Interface and expected behaviour only |
| RISC-V host | Interface and expected software flow only |
| `ACC_STORE` and `BREAK` ISA slots | No. These slots exist in the emulator only. |

## Requirement conventions

- Each requirement has an ID of the form `IPU-<AREA>-<n>`. Block specifications trace to these IDs.
- "Shall" marks a requirement on the IPU.
- `CU-<n>` marks a behaviour the IPU expects from the cache unit.
- `SW-<n>` marks a behaviour the IPU expects from host software.
- `F-<n>` marks a fault in the fault list.
- `OQ-<n>` marks an open question. `A-<n>` marks an action on another document.
- A row marked **Proposed** is not agreed. [Open questions](open-questions.md) holds its open question.

## Terms

| Term | Meaning |
|---|---|
| VLIW word | One instruction word. It holds one sub-instruction per slot. |
| Slot | A field of the VLIW word executed by one stage: `COND`, `LR` (three lanes), `LOAD`, `MULT`, `ACC`, `AAQ`, `STORE`. |
| Kernel | A program in one IMEM bank, with its CR bank and its table set. |
| Context | Context 0 or context 1. A context holds one IMEM bank, one CR bank and one table set. |
| Active context | The context the running kernel uses. The other context is the inactive context. |
| Locked | A locked context rejects host writes. [Execution flow](execution-flow.md#contexts) defines when a context is locked. |
| Armed | The inactive context is queued to run when the running kernel ends. |
| Retire | A word retires when the IPU has committed its last effect, including its store. |
| Done | A kernel is done when its `END` word retires. |
| Host | The RISC-V processor that loads kernels and controls the IPU. |
| Cache unit | The CRM, XMEM and DMA. It moves data between DRAM and XMEM. |
| XMEM bank | One of 16 on-chip memory banks. Each bank holds 1024 rows. |
| Table | A cache-unit data structure that maps one array to XMEM banks and DRAM. |
| Cache agent | The IPU stage that decides whether a word's memory accesses hit. |
| Hit, miss | A memory access hits when its XMEM bank is valid for the requested tag. Otherwise it misses. |
| Halt | The IPU stops at a word boundary and keeps its architectural state. |
| Debug cause | A halt cause other than a fault: halt request, step done, or breakpoint. |
| AXI error | A `SLVERR` response on the host port. |
| Fault | An illegal state that the IPU detects. |

## Source documents

| Document | Use |
|---|---|
| [Instruction reference](../instructions.md) | Instruction semantics and encodings |
| [Control stage specification](../specs/stage-control.md) | CTRL behaviour |
| [Accumulator stage specification](../specs/stage-accumulator.md) | ACC behaviour |
| [AaQ and store stage specification](../specs/stage-aaq-str.md) | AAQ and STORE behaviour |
| [Cache unit specification](../specs/cache-unit.md) | CRM, XMEM and DMA behaviour |

[Open questions](open-questions.md) lists the conflicts between these documents.
