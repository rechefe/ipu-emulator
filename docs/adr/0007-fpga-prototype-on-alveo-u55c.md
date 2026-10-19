# The FPGA prototype is a platform around a drop-in IPU, on an Alveo U55C

_Status: accepted as prototype defaults (2026-10-19). The specification is [FPGA prototype](../content/specs/fpga-prototype.md)._

The IPU is prototyped on an AMD Alveo U55C. This repository holds the platform around the IPU: the shell integration, a prototype cache unit, the host-port bridge, the platform registers and the host software. The IPU itself drops in from the RTL repository as a module `ipu_top` with the ports of [External interfaces](../content/requirements/interfaces.md). Until that RTL exists, a stub with the same ports stands in, so the platform can be brought up and tested alone.

## Decisions

The owner settled three of them (2026-10-19). The rest are the simplest choice that works, recorded here as prototype defaults; each can change without touching the instruction word.

| Decision | Choice | Settled by |
|---|---|---|
| Register maps | SystemRDL in this repository (`src/ipu-host/src/ipu_host/rdl/`). One file describes the IPU host port, one the prototype platform. PeakRDL generates the SystemVerilog register block, the Python driver's map and the docs page. | Owner |
| Instruction words in an IMEM window | One word per 32 bytes: word `i` is at byte `32 * i`. Beats 0 to 4 hold bits 31:0 to 159:128 of the 160-bit padded hardware word. Beats 5 to 7 read 0 and ignore writes. This answers the packing part of OQ-14. | Owner |
| First milestones | Platform only. The RTL repository has little IPU RTL yet. | Owner |
| Shell | Vitis user-managed RTL kernel on the XRT platform `xilinx_u55c_gen3x16_xdma_3_202210_1`, Vitis 2023.2. The x86 host reaches the kernel's AXI4-Lite control port over PCIe through XRT (`xrt::ip`). | Prototype default |
| Host | The x86 host stands in for the RISC-V host. It drives the same host port with the same register map. | Prototype default |
| Clock | One kernel clock, 100 MHz (`ap_clk`). The IPU's 1 GHz target applies to silicon only. | Prototype default |
| Device memory | HBM. The prototype cache unit's DMA uses one 512-bit AXI4 master on HBM pseudo-channel 0. | Prototype default |
| Cache unit | The prototype cache unit maps page `p` of a table to XMEM bank `first_bank + p` and never replaces a page. A table holds at most its banks. Streaming tables are a later phase. | Prototype default |
| Row address | `{table_id[3:0], offset[19:0]}` in the low 24 bits of `CR[base] + LR[offset]`, the proposal of OQ-8. | Prototype default |
| Element format | A `FORMAT` register per context in the host map (OQ-36). | Prototype default |
| Access rules | PeakRDL regblock answers every access with OKAY, so the IPU checks the host-access rules in front of the generated block and answers SLVERR itself. A generated address package gives it every offset. | Prototype default |
| Emulator programs | Flat mode: table 0, `scratch_pad`, banks 0 to 15 (2 MiB). Emulator row `r` is bank `r >> 10`, row `r & 1023`. Kernels and conformance cases run unchanged. | Prototype default |

## Consequences

- The RTL repository vendors `ipu_instr_pkg.sv`, as today, plus `fpga/alveo-u55c/rtl/ipu_socket_pkg.sv` and the files generated from `ipu_host.rdl` (`ipu_host_regs.sv`, `ipu_host_regs_pkg.sv`, `ipu_host_addr_pkg.sv`). Changing a register offset or field in `ipu_host.rdl` changes the RTL, so treat it like a `*_WIDTH` in the instruction package. Call the change out in the PR.
- The platform, the generated register blocks and the stub are elaborated with slang on every `bazel test //...`. Simulating them needs Verilator and runs outside CI (`fpga/alveo-u55c/sim/`). Synthesis needs a Vitis install and is never part of `bazel test`.
- The prototype runs only hardware element formats (INT8, E1 to E7) and programs assembled with `--target hardware`. On 2026-10-19 that covers the conformance suite's cases not marked `emulator_only`, and 9 kernel cases (the INT8 `proj_p4` family), which all depend on the AaQ placeholder (OQ-20). Most kernels declare FP32 only.
- The host software runs against the card (XRT) or against a model of the host port over the reference model. The same script runs in CI without a card.
