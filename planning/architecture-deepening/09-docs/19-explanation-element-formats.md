# Explanation: element formats, including CR scalar decoding

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-19 · Size M · Depends on: E09-05, E09-06, E03 (one element-format enum in `ipu_isa`, Q47), E04 (format codecs and handlers written once, Q48; CR scalar behaviour PR, Q49; numerics module) · Behaviour change: none

## Goal

A reader understands what an element format is, which formats exist on the
hardware and which only in the emulator, how each one encodes values, and how a
CR scalar operand is decoded in the active format.

## Background

- Today the format story is split across three mechanisms and three pages:
  - `IpuState.dtype` (`DType.INT8`, `E1` to `E7`), described in
    `ipu-configuration.md:21-48`;
  - wide-vector debug mode (`wide_vector_debug`, `wide_vector_arithmetic`
    FP32 or INT32; `ipu_state.py:42-96`), described in
    `wide-vector-debug-mode.md` (75 lines) as "emulator only";
  - `building-applications.md:59-62` repeats the wide mode.
- Q43, Q44, Q47, Q50: one element-format enum in `ipu_isa`: INT8 and E1 to E7
  (HARDWARE, selected at runtime) and FP32 and INT32 (EMULATOR). The format
  derives element bytes, row bytes, the accumulator type and registers. Wide
  mode is "run at full FP32": XMEM rows are 4× wider, a vector is always 128
  elements. Kernels are format-independent.
- Q49: CR scalar operands are decoded in the active format (the low
  element-width bits of the CR). Today wide mode reads the CR's low byte as a
  signed integer whatever the arithmetic (`ipu.py:693-696`, used at `:1248`,
  `:1348`, `:1379`), and narrow mode passes the low byte to `ipu_mult`
  (`ipu.py:1362`, `:1391`). E04 changes this in a behaviour PR after a kernel
  audit. The user asked for it to be documented.
- Q34: the FP8 special-value convention is open (OQ-19).
- Max finite values (plan 07, verified): E4 = 240, E5 = 57344, E3 = 15.5,
  E2 = 3.9375; E1 has no normal values.

## Implementation

1. Write `explanation/element-formats.md` with `anthropic-skills:technical-writing`:
   - What an element format is and what it determines (element width, row
     width, accumulator type), generated as a table from the `ipu_isa` enum.
   - HARDWARE and EMULATOR formats, and why FP32 and INT32 exist (simulating a
     kernel at full precision; Q43).
   - Encoding of each 8-bit format: bias, exponent and mantissa bits, max
     finite, smallest subnormal, rounding (ties to even), overflow
     (saturation), special values. Generated from the numerics module, with
     the open convention flagged and linked to OQ-19.
   - **CR scalar decoding** (Q49): a CR holds 32 bits; an instruction that
     takes a CR as a scalar element uses the low element-width bits,
     decoded in the active format. One table per format (which bits, how
     decoded) and one assembling example. State the version (the E04 PR) from
     which this holds.
   - How a kernel selects a format (case or `--format`, E06) and how XMEM
     layouts pack per format.
   - Arithmetic events (overflow, NaN) link to `requirements/faults.md` and
     OQ-18; AaQ quantization links OQ-20 and OQ-22.
2. Delete `wide-vector-debug-mode.md`. Remove the format text from
   `building-applications.md` (if it still exists) and from
   `ipu-configuration.md` (E09-14 removes the rest).
3. Link this page from the generated instruction reference wherever an operand
   is a CR scalar (E09-14).

## Tests

- [ ] The format tables are generated: adding a format to the enum adds a row
      with no doc edit.
- [ ] The CR scalar example is included from an emulator test that asserts the
      decoded value in at least INT8, E4 and FP32.
- [ ] `bazel test //docs:test_doc_asm_blocks` passes.

## Acceptance Criteria

- [ ] One page explains every element format; no other page repeats the
      format list.
- [ ] CR scalar decoding is documented with a tested example (Q49).
- [ ] The words "wide-vector debug mode" do not appear in `docs/content`
      except in the release log.
