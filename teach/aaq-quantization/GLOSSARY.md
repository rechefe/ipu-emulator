# AaQ Quantization Glossary

Terms for how the IPU's AaQ stage writes quantized rows to XMEM and how readers decode them. Names follow the repo specs.

## Rows

**Row**:
128 elements of 8 bits (1024 bits) at one XMEM row address, plus its metadata word.
_Avoid_: line, word, vector (for the stored form)

**Metadata word**:
The 16 bits stored beside every row: `scale[7:0]`, `dynamic_exponent[8]`, reserved `[15:9]`.
_Avoid_: header, row metadata row, sideband

**Scale**:
The row's shared exponent `s`, an `e8m0` value (bias 127): the largest raw FP32 exponent among the row's elements.
_Avoid_: scale factor (when the multiplier 2^(s−127) is meant, say so)

**`dynamic_exponent`**:
Metadata bit 8: the exponent split AaQ chose for a row of a dynamic table, 0 = E1M6, 1 = E2M5.

**Block scaling**:
Sharing one scale across a block of elements so each element stores only its offset from it. AaQ's block is one row; OCP MX uses 32 elements.

## Tables

**NX8**:
The hardware's 8-bit element family `e1`–`e7`: each element stores its distance below the row's scale. "E4" always means NX8 e4; the emulator's old biased E1–E7 are retired.
_Avoid_: AaQ E4M3, scaled E4

**`quantizion_format`**:
The CR operand that names the NX8 format (sign, `fe`, width, static or dynamic) in the architecture specs, on `ACTIVATE.QUANTIZE` and every MULT instruction. The AaQ and MULT layouts differ.

**Table format**:
The microarchitecture and cache-unit view: a 9-bit `format` property of a table, `{dynamic, sign, fe, W}`, shared by every row. The architecture specs use `quantizion_format` instead.
_Avoid_: row format

**`fe`**:
The number of exponent bits in an element, 1–7: from `quantizion_format` (the table format in the microarchitecture view), or from `dynamic_exponent` when the format is dynamic.

**Table set**:
The tables of one kernel, at most 16, held by the cache unit. There are two, one per context.

**Context**:
One IMEM bank, one CR bank and one table set, selected together by `active_ctx`.

**Cache agent**:
The IPU stage that decides hit or miss for each word's memory accesses, from its read-only copy of the table sets (`tbl_view`) and the bank tags.
_Avoid_: CRM (the cache-unit spec's older name for the tag check), cache unit (a different block)
