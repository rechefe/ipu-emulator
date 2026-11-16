# Memory sketch: how XMEM, banks, tables and the DMA fit together

One page for the memory what-if tool (`src/ipu-memsim`). Every statement is
labelled **CONFIRMED** (with the file and line it comes from, at base commit
`c5c04f7c`), **ASSUMED** (a default the tool picks; where to change it) or
**UNKNOWN** (a question in [questions.md](questions.md)).

Short paths: `CU` = `docs/content/specs/cache-unit.md`,
`MA` = `docs/content/requirements/memory-access.md`,
`OQ` = `docs/content/requirements/open-questions.md`,
`ARCH` = `docs/content/requirements/architecture.md`,
`EXEC` = `docs/content/requirements/execution-flow.md`.

## The picture

```text
 DRAM / HBM  <--- DMA --->  XMEM (16 banks)  <--- cache agent / CRM --->  IPU pipeline
 (all tensors)   background   (what is on chip      hit: word accepted        one word
                 copies       right now)            miss: stall CTRL          per cycle
```

- The IPU never talks to DRAM. It reads and writes XMEM rows. **CONFIRMED** (`MA:3`).
- A separate block, the cache unit (CRM + XMEM + DMA), copies whole banks
  between DRAM and XMEM. **CONFIRMED** (`CU:5-9`).
- The only thing that can slow the IPU down is a stall from the cache agent.
  **CONFIRMED** (`ARCH:107`, IPU-ARCH-14). Without a stall the IPU accepts one
  word per cycle (`ARCH:104`, IPU-ARCH-11).

So XMEM is **both** views the brief asked about: it is a cache in front of
DRAM (banks carry tags, a tag mismatch stalls), and a table of type
`scratch_pad` is a scratchpad that never goes to DRAM (`CU:87`). **CONFIRMED**.

## Units

| Thing | Size | Label |
|---|---|---|
| Row | 1024 bits = 128 bytes = 128 eight-bit elements | **CONFIRMED** `CU:48`, `CU:94` |
| Bank = page | 1024 rows = 128 KB | **CONFIRMED** `CU:48`, `CU:92` |
| XMEM | 16 banks = 2 MB | **CONFIRMED** `CU:47` (a parameter, `NUM_BANKS`) |
| Tables | up to 16 (4-bit `table_id`) | **CONFIRMED** `CU:50`, `CU:71` |
| Table offset | 20 bits = up to 1024 pages of 1024 rows | **CONFIRMED** `CU:49`, `MA:18-19` |

Example: row offset 2500 of a table is page `2500 // 1024 = 2`, row
`2500 % 1024 = 452` inside that page.

The emulator differs: its XMEM is a flat 512 MiB byte space
(`src/ipu-isa/src/ipu_isa/geometry.py:30`), not 2 MB, and has no banks, tags
or stalls (`OQ:63`, OQ-28). A row is 128 bytes in the 8-bit formats and 512
bytes in `FP32`/`INT32`, which are emulator-only formats
(`docs/content/explanation/element-formats.md:23-26`). **CONFIRMED**.
The tool counts in rows, so **ASSUMED**: one emulator row is one hardware
row, in any format (the trace header records the format).

## Tables

A table is one array (an input, a weight tensor, an output) plus the rules for
moving it (`CU:64-80`). **CONFIRMED** fields: `type` (`read_only`,
`write_only`, `read_after_write`, `scratch_pad`), `dram_base_address`, `size`,
`xmem_bank_list` with `xmem_num_of_banks`, `jump_back`, `repetitions`.

How a flat emulator row becomes `{table_id, offset}` is **UNKNOWN** on the
hardware side (`OQ:23`, OQ-8; the proposal there is that the base CR carries
the table ID). **ASSUMED** in the tool: each `kernel.toml` region (input,
output, scratch; `ipu_harness.placement.Region`) is one table, and the
offset is `row - region.base_row`. Change it in the recorder's region list
(`ipu_memsim.record`).

## Hit, miss and stall

- Page of an access = `offset[19:10]`; its bank is
  `banklist[page % number_of_banks_of_the_table]`. **CONFIRMED** (`CU:185-186`;
  the spec writes `% NUM_BANKS`, and the bank list of a table has
  `xmem_num_of_banks` entries, `CU:76-77`). This is a direct-mapped cache of
  pages, per table.
- Hit = that bank is valid and holds tag `{ctx, table_id, page}`.
  **CONFIRMED** (`MA:21`).
- A word is accepted only when all its accesses hit (`MA:39`, IPU-MEM-3); on a
  miss the cache agent stalls CTRL and sends one `bank_req` per missing tag,
  one per cycle, load first (`MA:41-42`, IPU-MEM-5/6). A stalled word has
  changed nothing (`ARCH:110-112`, IPU-ARCH-17). **CONFIRMED**. This is why
  replaying a recorded trace is legitimate: stalls delay the access sequence
  but cannot change it.
- A read bank is published only after the DMA filled the whole bank
  (`CU:153`, `MA:77` CU-21). **CONFIRMED**. So the first read of a page waits
  for up to 1024 rows of DRAM traffic, not one row.
- A write miss needs no DRAM read: the cache unit claims a free bank
  (`MA:59`, `MA:78` CU-22). **CONFIRMED**.
- How long a fill or a drain takes (DRAM latency, bandwidth, number of DMA
  channels) is **UNKNOWN**. **ASSUMED**: one DMA, a fixed latency plus rows
  divided by bandwidth (`[dram]` in the config).

## `jump_back`

Each table has one number, `jump_back`. When an access reaches row
`jump_back` or beyond inside page `p` (and `p != 0`), the kernel promises not
to touch page `p-1` again. For a read table the bank of `p-1` is released
(free); for a written table it is flushed (the DMA drains it to DRAM, then it
is free). **CONFIRMED** (`MA:31`, `CU:192-193`, `CU:225-227`, `MA:60-62`).

Example with `jump_back = 100`: the kernel reads rows 0..1023 of page 0, then
moves into page 1. At page-1 row 100 the bank holding page 0 is freed, and the
DMA can reuse it for page 2 (if the table has 2 banks, page 2 maps to the same
bank as page 0). `jump_back` is the look-back distance the kernel needs.

Who computes `jump_back` for a kernel, and what happens when a kernel
revisits a released page, is **UNKNOWN**. The tool's `cache_unit` model can
derive the smallest safe `jump_back` from the trace (**ASSUMED** default) or
take a fixed value.

## Is the DMA decoupled from the pipeline?

Yes. "DMA runs in the background" (`CU:115`), IPU and DMA "synchronize only
through per-bank tag/flush state" (`CU:107-108`), and the DMA may prefetch a
kernel's banks before that kernel starts (`EXEC:186`, CU-16). **CONFIRMED**.
Consequence for the tool: fixed-penalty models (AllMiss, Cache) are toy
models; the `cache_unit` model keeps an absolute clock so the DMA keeps
working while the IPU computes or stalls.

## Parameters the tool exposes

| Parameter | Hardware default | Config key |
|---|---|---|
| Number of banks | 16 (`CU:47`) | `num_banks` |
| Rows per bank (page size) | 1024 (`CU:48`) | `bank_rows` |
| XMEM size | `num_banks * bank_rows * 128 B` = 2 MB | derived |
| Row to bank mapping | per-table bank list, `page % n` | `mapping`, `bank_allocation` |
| `jump_back` per table | UNKNOWN | `jump_back` |
| Miss penalty (toy models) | UNKNOWN | `miss_penalty` |
| Cache size, line size, associativity (toy model) | n/a | `size_rows`, `line_rows`, `assoc` |
| DRAM/HBM latency and bandwidth | UNKNOWN | `[dram]` |
| Number of memory levels | 1 (XMEM) + DRAM | `[[levels]]` (design note, Stage 6) |
