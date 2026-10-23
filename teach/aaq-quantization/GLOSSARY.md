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

**Table format**:
The 9-bit `format` property of a table: `{dynamic, sign, fe, W}`. Shared by every row of the table and never written to XMEM.
_Avoid_: row format, element type (for AaQ tables)

**`fe`**:
The number of exponent bits in an element, from the table format (or from `dynamic_exponent` in a dynamic table).

**Table set**:
The tables of one kernel, at most 16, held by the cache unit. There are two, one per context.

**Context**:
One IMEM bank, one CR bank and one table set, selected together by `active_ctx`.

**Cache agent**:
The IPU stage that decides hit or miss for each word's memory accesses, from its read-only copy of the table sets (`tbl_view`) and the bank tags.
_Avoid_: CRM (the cache-unit spec's older name for the tag check), cache unit (a different block)
