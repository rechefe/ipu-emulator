# Debug a kernel

Find out why a kernel gives the wrong answer, faults, or runs slowly. Every
command runs from the repository root.

The debugger is the kernel runner in a terminal user interface (TUI). Add
`--config=debug` to `bazel run` on any kernel target. The run assembles the
kernel, loads its inputs and CRs, and stops at cycle 0, before the first
instruction word runs. It takes the same options as an ordinary run:

```bash
bazel run --config=debug //src/tools/ipu-apps:identity -- --case single_row
bazel run --config=debug //kernels/elementwise/scale -- --case one_row --format e4
```

A kernel with a `kernel.toml` (under `kernels/`) also takes `--debug` in place
of `--config=debug`. The debugger needs an interactive terminal.

Every key and mouse action is listed in the
[debugger controls](../reference/cli.md#debugger), generated from the
debugger's own key table. Press `?` in the debugger for the same list.

## Before you start

The debugger has four panes. Each pane's title starts with the number key
that focuses it:

| Pane | Shows |
|---|---|
| 1 Pipeline Registers | `R0`, `R1`, `R_C`, `R_MASK`, `R_ACC`, `POST_AAQ`, `MULT_RES`, `MEM_BYPASS` |
| 2 Instructions | The disassembly around the current PC, one instruction per row, with `||` marking the rest of a word's slots |
| 3 XMEM | A range of XMEM rows or bytes |
| 4 CR/LR | `LR0`–`LR15` and `CR0`–`CR15` |

Each pane has tabs. `F2` adds a tab and `F3` edits the active one; both open an
editor at the bottom of the screen. The value it takes depends on the pane:

| Pane | Tab request | Example |
|---|---|---|
| Instructions | `current`, or a PC from 0 to `INST_MEM_SIZE - 1` | `12` |
| CR/LR | `all`, `lr` or `cr` | `lr` |
| XMEM | `row|byte BASE OFFSET COUNT FORMAT` | `row CR3 0 128 f32` |
| Pipeline Registers | A register name | `R_ACC` |

In an XMEM request, `BASE` and `OFFSET` are each an immediate, an LR or a CR;
the debugger adds their values when it draws the pane, so a tab follows the
registers as they change. `row` addresses XMEM rows, `byte` addresses bytes.
`COUNT` is in values of `FORMAT`: bytes for `hex`, `int8` and `u8`, 16-bit
values for `cell16`, 32-bit values for `u32` and `f32`.

A value that changed since the last stop is bold and underlined.

## The output is wrong

A wrong output prints `FAIL` and the worst element:

```text
scale/one_row/e4: FAIL  cycles=7  MAC util=14.3%  y: FAIL at EXACT (max|err|=3.00195 max rel=1.00065); worst at [0, 10]: got -0.001953125, want 3.0
```

`want` is the [reference](../reference/glossary.md) rounded to the element
format. `worst at [0, 10]` is the output element (row 0, lane 10) with the
largest error.

1. Run the failing case, in the failing format, in the debugger:

    ```bash
    bazel run --config=debug //kernels/elementwise/scale -- --case one_row --format e4
    ```

2. Stop where the output is computed. Choose one:

    - Focus Instructions (`2`), move the cursor to the word, and press `F9` to
        set a breakpoint on it. A red `B` marks it. Press `F5` to run to it.
    - Move the cursor to the word and press `F10` to run to it once.
    - Add a `BREAK` instruction to the source, then rerun. `BREAK` stops every
        time its word issues; `BREAK.IFEQ` stops only when an LR holds a value,
        which stops a loop on one iteration:

        ```asm
        row_loop:
            BREAK.IFEQ LR1 3 ;
            LD.MULT R0 LR1 CR2 ;;
            BLT.S LR1 CR4 row_loop ;;
            HALT ;;
        ```

        `BREAK` is emulator-only: remove it before you assemble with
        `--target hardware`.

3. Press `F8` to run one instruction word at a time. The debugger stops before
    each word runs, so the panes show the state that word reads.

4. Check the scalars in CR/LR (`4`). Each region base and parameter is in the
    CR the kernel's [ABI](../reference/glossary.md) names (`[abi]` in
    `kernel.toml`). An LR that drives an address or a loop count must hold the
    value you expect.

5. Check the vectors in Pipeline Registers (`1`) and the data in XMEM (`3`).
    Press `F2` in XMEM and request the region, for example `row CR3 0 128 f32`
    for an output region whose base row is in `CR3`. Press `f` to cycle the
    tab's format until it matches the element format:

    | Element format | XMEM and `R0`, `R1` | `MULT_RES`, `R_ACC` |
    |---|---|---|
    | FP32 | `f32` | `f32` |
    | INT32 | `u32` (`int32` in Pipeline Registers) | `int32` |
    | INT8 | `int8` | `int32` |
    | E1–E7 | `hex` (the 8-bit codes) | `f32` |

    The debugger shows 8-bit float codes as bytes. [Element
    formats](../explanation/element-formats.md#how-the-8-bit-formats-encode-values)
    explains how to decode them, and the accumulator registers are already
    FP32.

6. Compare the values with the reference. To get the whole output as a file,
    run the case without the debugger and add `--output`. It writes the output
    even when the comparison fails: `--output DIR` writes `DIR/<name>.npy` for
    a kernel under `kernels/`, and `--output FILE` writes one file for a kernel
    target under `src/tools/ipu-apps`.

7. To test a fix without editing the source, press `e` on a value to change
    it, then continue. In Instructions, `e` sets the PC. `CR0` and `CR1` are
    read-only.

`q` or `Ctrl-C` quits. A quit run prints `QUIT  (no result)` and checks no
output. A run that reaches `HALT` closes the debugger and checks its outputs
as an ordinary run does.

## A fault halted the run

A fault stops the run at the faulting word and prints its code and PC:

```text
scale/one_row/e4: FAIL  cycles=6  MAC util=16.7%  fault F-2 at PC 6
```

A kernel target under `src/tools/ipu-apps` prints the same fault as
`error: F-2 at PC 6: ...`. The debugger also closes on a fault and prints this
line.

1. Look up the code in the [fault list](../requirements/faults.md#fault-list).
    [In the emulator](../requirements/faults.md#in-the-emulator) says what the
    emulator checks for each code.

2. Faults are precise: the faulting word changes no register and no XMEM byte.
    Run the case in the debugger and stop before the faulting word: press `F2`
    in Instructions, request its PC, and press `F9` on it, then `F5`.

3. Read the operands the fault checks. For example, F-4 checks the
    `LD.MULT.CYCLIC` index, and F-6 checks the XMEM row a load or store
    addresses (base plus offset, from CR/LR).

4. Some faults are reached rather than raised by the word at the PC:

    | Fault | Where to look |
    |---|---|
    | F-2 | The PC is past the program. The word before it fell through, or a branch jumped there; check for a missing `HALT` or a wrong branch target. |
    | F-9 | The run reached its cycle limit. A loop counter that never reaches its bound is the usual cause. `--max-cycles N` sets the limit for one run; 0 disables it. |

## The kernel is too slow

1. Read the run's statistics. A kernel target under `src/tools/ipu-apps`
    prints the [run statistics](../reference/cli.md#run-statistics) after
    `PASS`. A kernel under `kernels/` prints its cycle count and MAC
    utilisation on the result line:

    ```text
    scale/default/fp32: PASS  cycles=22  MAC util=18.2%  y: max|err|=0 (EXACT)
    ```

    `cycles` is the number of instruction words retired. `MAC util` is the
    multiplier work that was real arithmetic, as a fraction of the run's peak.
    Multiplies by a constant 1, which only move data, do not count.

2. Find the instruction sequences that stand in for missing instructions. Add
    `--profile-aliases` to the run:

    ```bash
    bazel run //kernels/elementwise/scale -- --case default --format fp32 --profile-aliases
    ```

    It prints one line per [ISA idiom](../isa-alias-catalogue.md): how often
    it occurred and how many cycles took part.

    ```text
    --- Full ISA measurements (overlapping; no savings estimates) ---
    A1_MOV_RC: no observed match
    A12_ADDRESSING/load_to_mult: 4 occurrences (verified), 8 participating cycles
    A13_NARROW_MULT/lanes_128: 4 occurrences (verified), 4 participating cycles
    ```

    The counts overlap: one word can take part in several idioms, so do not
    add them up.

3. For the full measurements (instruction sites, examples, lanes, XMEM bytes),
    write them as JSON with `--alias-report FILE` in place of
    `--profile-aliases`. For a kernel under `kernels/` run with
    `--format all`, each format writes `FILE.<format>.json`.

## Look at a recorded run in the browser

The [Cycle Inspector](../debugging-inspector.md) replays a recorded run cycle
by cycle, forward and back, with no terminal. It holds traces of a curated set
of kernels, and explains how to record a trace of your own case.
