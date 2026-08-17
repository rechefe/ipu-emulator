# IPU Bootcamp: three teaching drills

Three small, real, runnable IPU apps that build up the ISA surface one slot
at a time. Work through them **in order** -- each one assumes you understand
the previous one and explicitly calls out what's new.

| # | Drill | Difficulty | ISA surface added | Control flow |
|---|---|---|---|---|
| 1 | [`residual_add/`](residual_add/README.md) | * | LR, LOAD/XMEM | one counted loop |
| 2 | [`conv/`](conv/README.md) | ** | + MULT, ACC (real multiply-accumulate) | one counted loop |
| 3 | [`softmax/`](softmax/README.md) | *** | + AAQ (`ACTIVATE`), AGG reductions | four passes, staged through XMEM |

## Prerequisites

Read these two files before starting drill 1 -- everything below assumes you
already know this vocabulary:

- Repo root `CLAUDE.md` (build/test commands, `instruction_spec.py` as
  single source of truth, the "never assign opcodes manually" rule)
- Repo root `SKILL.md` (VLIW compound-instruction model, register file
  layout, slot list, snapshot-vs-live read semantics)

Each drill's `.asm` file also re-explains the specific rules it needs
in context (e.g. why a load has to happen one cycle before the multiply
that consumes it), so you don't need to hold the whole ISA in your head at
once -- just enough to follow along one drill at a time.

## Why this order

Every VLIW word is up to nine independent slots (`cond`, `lr` x3, `load`,
`mult`, `acc`, `aaq`, `store`, `acc_store`, `break`). Learning them all at
once from a production app (like `conv_first_layer` or `softmax_rows`,
which this bootcamp draws on but deliberately simplifies) means also
learning masking, BN/activation fusion, group loops, and stride decimation
in the same sitting. These three drills peel that apart:

1. **`residual_add`** uses only the LR and LOAD/XMEM slots. MULT and ACC
   appear, but only as plumbing (a "multiply by 1" trick to move data
   through the only arithmetic path the ISA has) -- there is no masking, no
   AAQ, and the control flow is a single counted loop. This is where you
   learn the read-before-write / snapshot rule and row-addressed XMEM, which
   every later drill depends on.

2. **`conv`** keeps the same LR/LOAD/XMEM skeleton from drill 1, but now
   MULT and ACC do real work: a 3-tap 1D convolution via the "shift is free"
   trick (`MULT.RC.VE`'s `rc_idx` operand shifts the whole 128-lane vector).
   Still no masking (edge lanes are simply not read back) and still one
   counted loop -- this drill isolates "real compute" from "real control
   flow," which drill 3 adds.

3. **`softmax`** keeps everything from drills 1-2 and adds the AAQ slot
   (`ACTIVATE.QUANTIZE`) plus `AGG.MAX`/`AGG.SUM` reductions, wired together
   into a genuine 4-pass kernel (max / exp / sum / normalize) with
   intermediate results staged through XMEM between passes -- the pattern
   nearly every non-trivial IPU kernel uses.

What's still **out of scope** even after all three drills: element masking
(`R_MASK`/`mask_shift` beyond their always-pass-through defaults), branches
other than a simple counted `BLT` loop, BN/activation fusion, and
multi-group/multi-tile chunking for inputs larger than one tile. Those all
show up in the project's full-scale apps once you're ready for them --
see project memory for `conv_first_layer`, `depthwise_conv_universal`, and
`softmax_rows_partial`/`softmax_rows_long` as the next step up.

## Running the drills

Each drill follows the same three-file pattern (`<name>.asm` + `__init__.py`
harness + `__main__.py` demo + `test_<name>.py`; see the reference app at
`src/tools/ipu-apps/src/ipu_apps/fully_connected/` for the general shape).
None of these are wired into Bazel yet (matching how several other
in-progress apps in this repo are currently un-wired) -- run them directly:

```bash
cd /path/to/ipu-emulator
PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
  python -m ipu_apps.bootcamp.residual_add --vectors 8

PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
  python -m ipu_apps.bootcamp.conv --rows 4

PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
  python -m ipu_apps.bootcamp.softmax --rows 16
```

And the tests:

```bash
PYTHONPATH="$(pwd)/src/tools/ipu-emu-py/src:$(pwd)/src/tools/ipu-common/src:$(pwd)/src/tools/ipu-apps/src:$(pwd)/src/tools/ipu-as-py/src" \
  python -m pytest src/tools/ipu-apps/test/test_bootcamp_residual_add.py src/tools/ipu-apps/test/test_bootcamp_conv.py src/tools/ipu-apps/test/test_bootcamp_softmax.py -v
```

Each drill's own README has a "try it yourself" exercise -- a suggested
modification or extension -- once the base version passes.
