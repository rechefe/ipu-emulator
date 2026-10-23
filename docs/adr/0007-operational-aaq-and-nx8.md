# Operational AaQ: NX8 rows with metadata, two run modes

_Status: accepted (2026-10-23). Supersedes the element-format list and the single per-run format of [ADR 0003](0003-element-formats-replace-wide-mode.md); amends [ADR 0002](0002-timing-contract.md) and [ADR 0005](0005-faults-follow-the-requirements.md) as stated below._

The emulator simulates the architecture specs (`ARCH-stage-aaq-str.md`, `ARCH-stage-mult.md`, `ARCH-stage-acc.md`) for everything a program can observe. Where the microarchitecture spec (`MAS-stage-aaq-str.md`) disagrees, the emulator follows the architecture spec and the difference is an open question (OQ-42).

A run is in one of two modes. **NX8** is the hardware: an XMEM row is 128 NX8 elements (128 bytes) plus a 16-bit metadata word (`scale[7:0]`, `dynamic_exponent[8]`), and every load and store moves exactly one row. **FP32** is emulator-only, for debugging kernels that will run in NX8: a row is 128 FP32 elements (512 bytes), with no metadata, no scaling and no quantization. `LD.MASK` and `LD.LUT` read the first 128 bytes of a row in both modes. INT8, INT32 and the emulator's own biased E1–E7 formats are removed; "E4" means NX8 e4 only. `ElementFormat` keeps two members, `FP32` and `NX8`.

An NX8 format comes from a 16-bit **QF** (quantization format) and the vector shape from a 16-bit **DS** (data structure), each held in one half of a CR (`CR13.L`, `CR13.H`). The layouts are provisional architect drafts (OQ-40):

| QF | Bits | DS | Bits |
|---|---|---|---|
| `num_bits` | [3:0] | `valid_elements` | [7:0] |
| `exp_format` (fe, 1–7) | [7:4] | `num_partitions` (1, 2, 4, 8 as 0–3) | [9:8] |
| `sign` | [11:8] | `mask_type` (0, +inf, −inf, NaN) | [11:10] |
| `dynamic` | [15:12] | `valid_partitions` | [15:12] |

A MULT constant CR holds `value[7:0]` (an NX8 element), `scale[15:8]` and an NX type (`fe[19:16]`, `sign[23:20]`, always 8 bits wide, never dynamic). `CR0` and `CR1` read as 0.0 and 1.0 in both modes. LR and COND operands accept full CRs and sign-extended halves.

The AaQ stage stores its own result: `ACTIVATE.QUANTIZE activation_fn, ds, qf, table_id, table_offset` activates, quantizes and writes one row with its metadata; `LD.LUT lut_addr, table_id, table_offset` loads one 1024-bit LUT segment from XMEM (the specs say from `R_ACC`; OQ-39). `ACTIVATE.EXACT` is an emulator-only instruction in its own slot that applies the exact Python activation and stores an FP32 row; it is valid in FP32 mode only. `ST.AAQ`, `ST.ACC`, `POST_AAQ_REG`, the STORE slot and the ACC_STORE slot are removed. `LD.MULT` and `LD.MULT.CYCLIC` copy a row's metadata into `R0`, `R1` or the `R_CYCLIC` slot, and MULT reads that metadata at cycle start (an addition to the forwarding table of ADR 0002).

Exceptions to following the architecture spec: inactive AaQ elements are written as 0, because NX8 has no NaN (OQ-36); `LD.LUT` reads XMEM (OQ-39); row metadata has no `dynamic_static` bit (OQ-44). ADR 0005's fault list is unchanged. Contents and inputs the specs leave undefined raise an emulator error at one of three points. Invalid QF, DS or constant contents are added to ADR 0005's "configuration error" kind and raised in the word's check phase, so the word leaves the state as a fault does; a row outside XMEM is fault F-6, found in the same phase. A program that uses `ACTIVATE.EXACT` is refused when it is loaded into an NX8-mode state. A NaN to quantize in NX8, `exp2` of a positive input and `generic` with no declared function depend on `R_ACC`, which the same word's ACC forwards, so they are emulator errors raised when `ACTIVATE.QUANTIZE` executes: the word's earlier writes (LR, LOAD, MULT, ACC) have already landed.

The reference-model API takes a major version bump to 2.0.0, despite [ADR 0004](0004-stable-reference-model-api.md)'s non-breaking promise: a caller that names INT8, INT32 or an E-format cannot keep working once those formats are gone.
