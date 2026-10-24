# Timing contract: cycle start for Control, forwarded values for the datapath

_Status: accepted (2026-10-15). Implementation is planned in `planning/architecture-deepening/`. Amended by [ADR 0007](0007-operational-aaq-and-nx8.md): MULT also reads the row metadata of R0, R1 and R_CYCLIC at cycle start._

Within one instruction word, the Control stage (LR and COND instructions) reads LRs at cycle start, and every other stage reads LRs after this cycle's LR writes. MULT reads R0, R1, R_CYCLIC and R_MASK at cycle start. MULT_RES is forwarded to ACC, AGG and RESHAPE, and R_ACC to AAQ, in the same cycle. The rules are declared per instruction in `ipu_isa` (slot execution order plus implicit reads) and the emulator is tested against them.

This follows the RTL control spec and replaces per-handler choices that disagreed with each other. It changes AGG `dest_slot` and RESHAPE `source`/`dest` from cycle start to forwarded, which requires fixing three attention kernels.
