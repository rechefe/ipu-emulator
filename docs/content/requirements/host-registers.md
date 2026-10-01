# Host registers

The host reaches the IPU through the host port, one AXI4 slave. The IPU block specification assigns register offsets and field layouts.

## Address spaces

| Space | Content | Write | Read |
|---|---|---|---|
| IMEM window 0 | IMEM bank 0 | While context 0 is not locked | In every state, except a running state with context 0 active |
| IMEM window 1 | IMEM bank 1 | While context 1 is not locked | In every state, except a running state with context 1 active |
| Register file | The register groups below | Per register | Per register |

The width of the hardware VLIW word and its packing into 32-bit beats are open: [OQ-14](open-questions.md).

## Requirements

| ID | Requirement |
|---|---|
| IPU-HOST-1 | The host port shall be one AXI4 slave with 32-bit data. |
| IPU-HOST-2 | The host port shall decode three address spaces: IMEM window 0, IMEM window 1 and the register file. |
| IPU-HOST-3 | The IMEM windows shall accept INCR bursts. |
| IPU-HOST-4 | The IPU shall check the access rules on each beat of a burst. |
| IPU-HOST-5 | An illegal access shall return an AXI error. It shall change no state except the `HOST_ERR` interrupt status bit. |
| IPU-HOST-6 | A host access shall never stall or disturb the pipeline. |
| IPU-HOST-7 | The IPU shall drive one level-sensitive interrupt, `irq`. `irq` is high while any interrupt status bit with its enable bit set is high. |
| IPU-HOST-8 | The host shall clear an interrupt status bit by writing 1 to it. |

[Faults](faults.md#illegal-host-accesses) lists the illegal accesses.

## Register groups

| Group | Content | Access |
|---|---|---|
| Identification | Block ID and version | Read-only |
| Command | `ARM`, `DISARM`, `HALT`, `STEP`, `RESUME`, `ABORT` | Write 1 to one bit to issue a command |
| Status | Run state, `active_ctx`, armed flag, `tables_valid`, `ctx_locked`, halt cause, breakpoint index, fault code, fault PC | Read-only |
| Interrupt status | One bit per interrupt event | Read, write 1 to clear |
| Interrupt enable | One bit per interrupt event | Read and write |
| CR bank 0 | `CR0`–`CR15` of context 0 | Read. Write while context 0 is not locked. `CR0` and `CR1` are read-only. |
| CR bank 1 | `CR0`–`CR15` of context 1 | Read. Write while context 1 is not locked. `CR0` and `CR1` are read-only. |
| Fault control | One enable bit and one sticky flag per fault | Read and write. Flags: write 1 to clear. |
| Program length | One register per IMEM bank. **Proposed**, for fault F-2. | Read. Write while the context is not locked. |
| Limits | Stall limit and kernel cycle limit. **Proposed**, for faults F-8 and F-9. | Read and write |
| Debug window | Capability register, PC, `LR0`–`LR15`, breakpoint registers | See [Debug](debug.md) |

## Interrupt events

| Event | The IPU sets the status bit when |
|---|---|
| `KERNEL_START` | A kernel starts. |
| `KERNEL_DONE` | A kernel is done. |
| `HALTED` | The IPU enters the `HALTED` state, for any cause. |
| `HOST_ERR` | The IPU rejects a host access. |

At a handoff, the IPU sets `KERNEL_DONE` and `KERNEL_START` in the same cycle.
