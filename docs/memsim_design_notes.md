# Memory what-if tool: design notes for what comes next

Design only; nothing on this page is implemented. It covers three
extensions: a second memory level, a second chip, and validation on the FPGA.

## How the tool is put together today

```text
kernel --record--> trace (file) --replay--> MemoryModel --> stall cycles
                                              |
                                  CacheUnit --+--> DramBackend (fixed / table from DRAMsim3)
```

Three interfaces, each one small:

| Interface | Methods | Lives in |
|---|---|---|
| `MemoryModel` | `start(trace)`, `access(t, op, row, region) -> stall`, `finish(t)`, `counters()` | `ipu_memsim/models.py` |
| `DramBackend` | `transfer_cycles(rows, write) -> cycles` | `ipu_memsim/dram.py` |
| trace | five columns + a header | `ipu_memsim/trace.py` |

The extensions below add implementations of these interfaces. None needs a
change to the replay loop or to the trace format.

## A second memory level

Today there are two levels: XMEM (on chip) and DRAM/HBM. A middle level could
be, for example, a larger and slower on-chip or on-package memory between
XMEM and HBM.

Where it plugs in: **behind the `DramBackend` interface.** The `cache_unit`
model asks "how long does it take to bring these rows in?" and does not care
what answers. A `LevelDram(DramBackend)` would hold its own state:

```python
class LevelDram(DramBackend):
    def __init__(self, size_rows, hit_cycles_per_row, next_level: DramBackend): ...
    def transfer_cycles(self, rows, write):
        # rows present in this level cost hit_cycles_per_row each;
        # the others are fetched from next_level first.
```

One interface change is needed, and it is small: `transfer_cycles(rows,
write)` must also receive *which* rows (table and page), because a level with
state has to know whether it holds them. `CacheUnit._dma` already has the
table and the page at hand. In the config a chain of levels would read:

```toml
[[levels]]            # nearest to XMEM first
kind = "level"
size_rows = 262144    # 32 MB
hit_cycles_per_row = 0.25

[[levels]]
kind = "table"        # HBM, measured with DRAMsim3
fill = { "1024" = 2111 }
```

Example of the question it answers: with 2 MB of XMEM and a 32 MB middle
level, weights that are re-read for every layer come from the middle level
(say 256 cycles per bank) instead of HBM (2111 cycles per bank).

The number of levels is then the length of the `[[levels]]` list, which is
the "number of levels" knob the team lead asked for.

## A second chip

Two IPUs running parts of one network (or two networks) at the same time.
What changes is that some resource is shared: the DRAM/HBM and its DMA
bandwidth, and possibly a chip-to-chip link.

Plan, in two steps:

1. **Record per chip.** Each chip runs its own list of kernels, so it has its
   own trace. Nothing new: a "chip" is a network file (see
   `networks/mobilevit_s.toml`) and the traces of its layers, concatenated in
   time (`cycle` shifted by the cycles of the layers before it). A helper
   `concat(traces)` is the only new code on the trace side.
2. **Replay together.** A `replay_many(traces, models, shared)` loop advances
   several replays in time order: at each step it takes the access with the
   smallest current time `cycle + shift` among the chips. Each chip keeps its
   own `MemoryModel` (its own XMEM), and they share one `DramBackend` object.
   For sharing to mean anything, the backend must be told *when* a transfer
   is requested and answer when it completes
   (`transfer(begin, rows, write) -> end`), so that chip B's transfer queues
   behind chip A's. Today the DMA queue lives in `CacheUnit._dma`; this step
   moves that queue into the backend. It is a move of about 10 lines, not a
   rewrite.

A tensor that chip A produces and chip B consumes becomes a dependency
between two replays: B's first read of that table may not happen before A's
last write of it plus a link delay. In the merged loop this is one extra
rule ("this table is ready at time T"), stored in the shared object.

Open questions for this part are Q18 in `questions.md` (shared HBM or one
each, the link, who schedules).

## What to compare on the FPGA

The purpose is to find out whether the model's stall counts are right, and to
replace placeholder numbers by measured ones.

### What is needed from the FPGA (ask the FPGA owner, Q15)

| Wanted | Why |
|---|---|
| A cycle counter from kernel start to kernel done | The one number the tool predicts: `total_cycles`. |
| A stall-cycle counter (cycles CTRL was stalled) | `stall_cycles`, separately from compute. |
| Per-table or per-bank counters: `bank_req`, `bank_pub`, `bank_release`, `bank_flush` | The model's `fetches`, `claims`, `drains`, `conflicts`. |
| Time from `bank_req` to `bank_pub`, for a full bank | Calibrates the DRAM backend (Q1) directly. |
| The table setup used (banks per table, `jump_back`) | The model's inputs; without them a comparison means nothing. |
| Whether bank count, DMA rate or `jump_back` can change without a rebuild | Decides how many points of a sweep can be measured. |

### Test plan

Each step lists what is run, what is compared and what a mismatch would mean.

1. **Compute-only check.** Run a small kernel whose data fits one bank per
   table with everything preloaded (`residual_add`, 64 channels: 325
   cycles). Compare the FPGA cycle count with `ideal_cycles`. A mismatch
   here is not about memory: it is pipeline depth or start-up cycles, and
   becomes a constant to add.
2. **One bank fill.** Same kernel, cold. The stall should be one fill per
   input table. This measures the `bank_req` to `bank_pub` time for a
   partial bank (64 rows) and, with a 1024-row input, for a full bank. Put
   the two numbers in a `[dram] kind = "table"` config.
3. **Streaming kernel.** `pointwise_conv_unified` at 128x128, 64 to 32
   channels: 8 input pages, 4 output pages, 316,553 ideal cycles. Compare
   total stall cycles and the number of fetches and drains with the
   `cache_unit` replay under the measured table. This tests prefetch,
   `jump_back` release and background drain.
4. **Re-read kernel.** A matmul whose weights are read once per output row
   (`matmul_144x144_x128`). Run it with enough banks for the weights, then
   with too few. This tests what the model calls `conflicts` and
   `refetches` (question Q5): the place where the model is most likely
   wrong, because the specification has no eviction.
5. **Sweep one knob.** If the FPGA can change the bank count or the banks per
   table without a rebuild, repeat step 3 at 3 or 4 settings and overlay the
   measured points on `bank_count.svg`.

Agreement to aim for: the fetch, claim and drain *counts* should match
exactly (they are deterministic); stall cycles within a few percent once the
fill time is calibrated. A count mismatch points at a wrong assumption (list
in STATUS.md), not at a parameter.
