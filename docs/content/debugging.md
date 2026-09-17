# Debugging IPU Programs

The IPU emulator includes a powerful interactive debugger that allows you to pause execution, inspect registers, step through instructions, and modify state at runtime.

## Quick Start

Use native Bazel commands with a registered kernel target:

```bash
bazel run //src/tools/ipu-apps:softmax_rows -- --rows 8
bazel test //src/tools/ipu-apps:test_softmax_rows
bazel run --config=debug //src/tools/ipu-apps:softmax_rows -- --rows 8
bazel run --config=debug //src/tools/ipu-apps:identity -- --case single_row
bazel run --config=debug //src/tools/ipu-apps:fully_connected
```

`--config=debug` sets `IPU_DEBUG_TUI=1` for the launched application through
`.bazelrc`. The shared harness opens the TUI at cycle 0 after loading the
program, inputs, and registers, before any instruction executes. Every harness
using `IpuApp.run()` supports this; no BREAK instruction, per-kernel launch
code, or debugger import is required. Case selection and arguments are the
same as an ordinary run.

Press `F8` to step, `F5` to continue, or `F9` in disassembly to toggle a
breakpoint. `F10` runs to the selected instruction.
In Disassembly, press `f` to cycle compact, full, and stages views.
Debug sessions stay in the TUI. `q` or `Ctrl-C` cancels the application,
restores the terminal, cleans temporary files, and exits successfully without
writing or checking incomplete output. Completion runs the normal teardown
and case checks.

All seven app targets support this configuration: `fully_connected`,
`identity`, `softmax_rows`, `softmax_rows_partial`, `softmax_rows_long`,
`softmax_columns`, and `softmax_columns_packed`. Use ordinary Bazel labels;
inside the apps package, `:identity` is also valid. Run each suite with its
`test_<kernel>` label. There is no custom Bazel command or repository wrapper.

An interactive input and output terminal is required. Terminal initialization
failures abort the launch. Plain `bazel run` and `bazel test` remain
noninteractive when the debug environment flag is absent. For case options,
use `bazel run //src/tools/ipu-apps:identity -- --help`.

### Deprecated line debugger

The old line debugger is deprecated and retained for existing Python callers.
It is not accessible from the TUI. Existing callbacks continue to work:

```python
from ipu_emu.emulator import run_with_debug
from ipu_emu.debug_cli import debug_prompt

run_with_debug(state, lambda s, c: debug_prompt(s, c, level=1))
```

This lower-level API enters the line debugger when a BREAK instruction or
runtime breakpoint fires. Pass `break_on_entry=True` to stop before execution.

## Break Instructions

### Unconditional Break

```asm
break ;;
```

Always halts execution and enters the debug prompt.

### Conditional Break

```asm
break.ifeq lr0 5 ;;
```

Halts execution only when `lr0` equals `5`. Useful for breaking on specific loop iterations.

### No-Op Break

```asm
break_nop ;;
```

Does nothing (placeholder). Used when you want an explicit break slot but no action.

## Debug Levels

For direct `debug_prompt` callers, the `level` argument controls verbosity:

| Level | Description |
|-------|-------------|
| 0 | Print LR registers only |
| 1 | Also print disassembled current instruction (default) |
| 2 | Automatically save registers to JSON file |

These levels affect the line debugger; `bazel run --config=debug` opens the TUI directly.

## Debug Commands

### Runtime Breakpoints and Stepping

`break [PC]` adds a breakpoint at a numeric PC, defaulting to the current PC.
`breaks` lists runtime breakpoints; `delete PC` removes one and `delete all`
removes all of them. PCs accept decimal and prefixed integer literals such as
`0x10`, and must be within instruction memory. These breakpoints are retained
for the current IPU state and do not modify the program's BREAK instructions.

`until PC` resumes until that PC or another debugger stop, whichever happens
first. The temporary target is cleared at any stop or termination. If the
target is already the current PC, execution stays paused.

Every stop is before instruction side effects. `step` executes exactly one
VLIW instruction, then stops before the next instruction unless the program
has halted. Cycle counts report completed instructions. Continuing past a
breakpoint executes its instruction once; revisiting it in a loop stops again.
Simultaneous stop causes share one prompt and appear together in the TUI header.

### Persistent Full-Screen View

Launch the curses interface directly with:

```bash
bazel run --config=debug //src/tools/ipu-apps:identity
```

The deprecated `debug_prompt` entry point remains CLI-only. The TUI keeps disassembly, LR/CR,
XMEM, and pipeline registers visible together without embedding the normal
debugger command line. Each pane has its own tabs, cursor, scroll position, and
display format. Duplicate tabs are allowed.

Fresh sessions start with these tabs:

- Instructions: current-PC compact, full, and stages views.
- CR/LR: all, LR, and CR, each in hex.
- XMEM: row 0 in hex and a numeric view (f32 for wide floating-point, u32 for
  wide integer mode, int8 otherwise).
- Pipeline Registers: R0, R1, R_C, R_MASK, R_ACC, POST_AAQ, MULT_RES, and MEM_BYPASS.

Pipeline defaults match storage: mask bits; signed bytes for native INT8 data;
f32/int32 for wide arithmetic; int32 for native accumulator/multiplier results;
hex for bypass and FP8 raw bytes. POST_AAQ uses bytes when quantized and f32/int32
when wide output is unquantized. Tabs have explicit formats that remain editable.
Closing or changing tabs is preserved across steps; defaults are not re-added.
Closing the final Instructions, CR/LR, or Pipeline tab creates one fallback tab.
XMEM may have no tabs.

Default rows split usable terminal height evenly. Pipeline Registers takes
roughly half the width; CR/LR starts at 40–50% depending on its grid. Minimum
sizes still apply, and manual split sizes survive terminal resizing.

Disassembly tabs follow either the executing PC or a fixed numeric PC. Their
formats are `compact`, which removes unused NOP operations, and `full`, which
shows every operation executed in parallel. An instruction never runs sideways
across the pane: its operations are stacked one per row under its program
counter, and every row after the first is marked `||`, the way VLIW assembly
listings mark a slot issued in parallel with the one above it. Wide panes align
the operands; narrow panes reduce mnemonic padding to expose more of each
operation. The `||` markers use the dim border colour. Because the
rows of one instruction belong together, the cursor and the executing-position
marker cover all of them. LR/CR tabs select `all`, `lr`, or
`cr` and support `hex`, `u32`, `int32`, `f32`, and `bits`.

XMEM tabs retain their symbolic requests, such as `row cr4 lr0 16 f32`, and
resolve LR/CR values again whenever the display refreshes. Their formats cycle
through `hex`, `int8`, `u8`, `cell16`, `u32`, and `f32`. Changing format keeps
the exact resolved byte address and byte count; formats incompatible with the
range's alignment or size are skipped.

Pipeline tabs select `R0`, `R1`, `R_C`, `R_MASK`, `R_ACC`, `POST_AAQ`,
`MULT_RES`, or `MEM_BYPASS`. They support `hex`, `int8`, `u8`, `cell16`, `u32`,
`int32`, `f32`, and `bits`; the initial automatic format follows the active
execution mode and arithmetic type. Their value formatting matches XMEM:
`u8` and `u32` are zero-padded, `cell16` preserves its encoded colors and
two-character grouping, and wide-mode `hex` separates each four-byte group
with an extra space. The `bits` views show exact 32-bit patterns. LR/CR `u32`
values use the same zero padding. Only the focused pane uses the bright
selected-tab color; active tabs in other panes remain visible without
competing with the focus.

Emphasis is layered so that no two signals collide. A value changed since the
previous stop is bold and underlined, the cursor under it is reverse-video, and
the focused pane is the one with the bright frame, the bright title, and the
reverse-video active tab; the shortcut bar names it as well.

Every pane shows where its own cursor is, so switching panes never loses the
place: the focused pane draws its cursor in the accent colour and the others
draw theirs in a muted one. Everything that
carries no information recedes to the dim border colour: zero register, XMEM,
and pipeline values, the leading zeros of any wider zero-padded value, and
disassembly rows for instruction slots that hold no instruction. The executing
instruction keeps its marker and stays bright.

A pane title carries its `1`–`4` focus key, the pane's name, and what its tab bar cannot say - the
resolved XMEM address and byte count, or the pipeline register's effective
format and size. It does not repeat the active tab's label.

The header highlights `PAUSED` and shows the PC, completed cycle count, and stop
reason. A green arrow marks the current PC, a red `B` marks a runtime breakpoint,
and the selection has its own background. The symbols remain distinct in
monochrome terminals. The footer keeps Run, Step, and Quit visible
at 80 columns, with additional controls appearing as space allows.

The view remains active after stepping and at later breakpoints. `q` quits
execution, except while the help overlay is open, where it closes the overlay.
Outside an editor and the help overlay, `Esc` does nothing.

| Key | Action |
|-----|--------|
| `F5` | Continue to the next breakpoint |
| `F8` | Execute one instruction and refresh |
| `F9` | Toggle a runtime breakpoint at the disassembly cursor |
| `F10` | Run to the disassembly cursor or the next stop |
| `e` | Edit the selected LR/CR, pipeline-register, or XMEM value |
| `q` | Halt execution and exit the debugger (ignored while Help is open) |
| `Esc` | Close the help overlay or cancel an open editor; otherwise do nothing |
| `F2` | Add a tab to the focused pane |
| `F3` | Edit the active tab |
| `F4` | Close the active tab |
| `d` / `a` | Select the next/previous tab in the focused pane |
| `f` | Cycle the active tab's display format |
| `Tab` | Select the next tab in the focused pane |
| `Shift-Tab` | Focus the next pane |
| `1` - `4` | Focus Pipeline Registers, Instructions, XMEM, or CR/LR respectively |
| Arrow keys | Move the cursor inside the focused pane |
| `Page Up` / `Page Down` | Move the focused pane's cursor by one page |
| `Home` / `End` | Jump to the first/last value in the focused pane |
| `?` | Open the help overlay |
| `Shift-Left` / `Shift-Right` | Move the split on the focused row |
| `Shift-Up` / `Shift-Down` | Move the top/bottom split |
| `=` | Reset every split |
| Left / Right, Home / End | Move within an open tab editor |

Shifted arrow keys depend on the terminal reporting them. `xterm-256color`,
`screen-256color`, and tmux do; a plain Linux console does not, where those keys
do nothing.

Disassembly marks runtime breakpoints with `B` beside the current-PC arrow;
cursor highlighting remains independent of both markers. `F9` and `F10` apply
only when disassembly has focus.

The console uses one header row and one footer row. The header shows the kernel name
and a clickable **[? Help]** button at the top right. Status and error messages appear immediately left of Help in the header.
The footer contains shortcuts only, grouped with `|` separators: execution, navigation, view/edit,
layout, and exit. Keys 1–4 still switch panes but have no bar hints. Resize pane
dividers to reveal more of truncated values.

In the LR/CR pane, `e` opens a scalar-value editor. Enter a decimal or prefixed
integer (for example, `42`, `0x2a`, or `0b101010`); the emulator's normal register
masking applies. CR0 and CR1 remain read-only. Invalid values remain in the
editor with an error; `Esc` cancels without writing. Execution shortcuts are
inactive in the editor. Successful edits immediately refresh symbolic XMEM
tabs that reference the changed register.

Steps and breakpoint stops remain in the TUI. Tabs, selection, breakpoints,
and pane sizes are retained in memory for the current IPU state.

`?` opens a scrollable overlay listing every control, grouped by scope. While it
is open it consumes all input, so `F5` and `F8` cannot start execution by
accident; `Esc`, `?`, or `Enter` closes it. The overlay and the shortcut
bar are generated from one table in `debug_tui.py`, so they cannot disagree.

The bottom shortcut bar changes with the focused pane, so it only shows actions
that apply to the current context. It drops the least important shortcuts on a
narrow terminal rather than truncating a label, so `q Quit` remains
readable; the top-right **[? Help]** button lists whatever the bar had to leave
out. The header also identifies the kernel and execution state.

`F2` and `F3` open the same bottom editor with syntax determined by the focused
pane:

| Pane | Editor value |
|------|--------------|
| Disassembly | `current` or a PC from `0` through `INST_MEM_SIZE - 1` |
| LR/CR | `all`, `lr`, or `cr` |
| XMEM | `row|byte BASE OFFSET COUNT FORMAT`, for example `row cr4 lr0 16 f32` |
| Pipeline | A logical name such as `R0`, `R_ACC`, or `MEM_BYPASS` |

The editor validates its complete value before adding or replacing a tab.
Errors remain visible without closing it. Apart from the documented single-key
bindings `q`, `f`, `a`, `d`, `e`, `1`–`4`, `?`, and `=`, printable input outside an editor is
ignored, so accidental typing cannot execute debugger commands.

The clickable pane selectors at the top stay visible in every layout.
Zoom/Restore and Help buttons appear beside them when space permits. Footer
shortcuts are clickable too, including Run, Step, Zoom, Help, and Quit. Clicking
a value focuses its pane and places the cursor on it; clicking the first two
columns of an instruction row toggles its breakpoint. Double-click an LR or CR
value to open its editor; slow clicks only select it. Tab labels, close buttons,
overflow arrows, and each pane's add-tab button also accept clicks. Long active
tabs shorten to fit narrow panes, keeping their close button accessible.

Drag a scrollbar or click its track to jump through the content. Grabbing the
thumb preserves the selection until the pointer moves, including when releasing
it without dragging. The wheel moves
three lines within the pane under the pointer; Shift-wheel moves sideways.
Scrolling over a tab bar selects the previous or next tab. Drag a shared pane
border to resize it. A keystroke or terminal resize cancels an unfinished drag.
Help accepts wheel scrolling and a click on its close button. In an editor,
click within the input to position the cursor, or click Apply or Cancel.
Overlays consume mouse events so controls behind them cannot be activated.
Keyboard controls remain available when the terminal does not report mouse events.

Pointer motion is reported for every cell the pointer crosses, which is far
faster than a frame can be drawn, so the view keeps redrawing cheap. Instruction
memory and XMEM are stable while the TUI is paused, so decoded instructions,
resolved XMEM reads, and formatted XMEM lines are cached within each entry.
Scalar edits invalidate the XMEM caches so symbolic addresses are resolved
again. Motion that changes nothing costs no frame and cannot cancel a pending
redraw. Queued events are consumed individually, stopping as soon as execution
resumes, so a
burst of drag events costs one frame instead of one each.

The TUI uses a four-pane layout at 80 columns by 24 rows and larger:
1 Pipeline Registers at top left, 2 Instructions at top right, 3 XMEM at bottom
left, and 4 CR/LR at bottom right. The footer shows shortcuts without a focused-pane label; action/error messages
appear in the header. Below that,
a compact layout shows the focused pane; use keys 1–4 to
switch panes. The minimum usable size is 48 columns by 12 rows. Smaller windows
show a resize notice, and growing the terminal restores the view. Saved split
sizes survive temporary shrinking, including transitions through compact mode.

The input loop checks the actual terminal dimensions every 100 ms as well as
handling resize events, so resizing redraws immediately even when ncurses misses
the notification. It clears the previous screen before repainting. If curses
cannot initialize, the debug run is cancelled and the terminal is restored.
The header reports actions and errors to the left of Help; the footer shows
the main shortcuts that fit the available width.

Every pane reports its visible range on a border: the XMEM and pipeline panes
use their bottom border, and the disassembly and LR/CR panes use the right end
of their own top border, because their bottom border is shared with the pane
below. A pane title drops whole trailing segments on a narrow terminal rather
than cutting one in half, and gives up its readout before its own name.

The disassembly, XMEM, and pipeline panes each reserve their right-most content
column for a scroll thumb, so their frames stay unbroken and the layout does not
reflow when the thumb appears. The LR/CR pane scrolls sideways instead: it shows
as many whole register columns as fit, keeps at least two spaces between them,
and reports `columns 2-4/4` when the cursor moves past the visible ones. The tab
bars mark tabs scrolled out of view at either end.

Panes are drawn as one continuous frame: adjacent panes share a border, and the
shared cells are resolved into the right junction character instead of being
overwritten. Box-drawing characters are used when the terminal encoding accepts
them and a plain ASCII set is substituted when it does not.

Both splits are adjustable and persist across steps and later breakpoints.
`Shift-Up` and `Shift-Down` move the top/bottom split; `Shift-Left` and
`Shift-Right` move the split on whichever row holds the focused pane, so the
same two keys size the disassembly/LR-CR split and the XMEM/pipeline split.

The LR/CR grid reflows to the height it is given. Every shape divides a group's
16 registers exactly - 16 rows in one column, 8 in two, or 4 in four - and the
tallest shape that fits wins, because it also needs the fewest columns. The
default top-pane height is exactly what the tallest shape that fits the terminal
needs, so the grid never leaves dead rows, and a taller pane hands width back to
the disassembly as well: on a terminal with room for the 16-row shape the LR/CR
pane shows all 32 registers in about half the width the 8-row shape needs.
Without an override the pane takes the width its current shape needs but never
more than half the terminal; when even that is too narrow the grid scrolls its
columns and says so. The help and editor overlays are centred between the header
and the status bar and never cover either.

### Navigation

| Command | Description |
|---------|-------------|
| `continue` / `c` | Continue execution until next break |
| `step` | Execute one instruction, then break again |
| `quit` / `q` | Halt execution and exit |
| Up / Down Arrow | Move through command history in an interactive terminal |

### Register Inspection

| Command | Description |
|---------|-------------|
| `regs` | Print all registers |
| `lr` | Print all LR registers (loop registers) |
| `cr` | Print all CR registers (control registers) |
| `pc` | Print program counter |
| `r` | Print R registers (mult stage) |
| `rcyclic` | Print R cyclic register (512 bytes) |
| `rmask` | Print R mask register |
| `acc` | Print accumulator register |

### Reading Specific Values

```bash
# Get a single register value
debug >>> get lr0
lr0 = 128 (0x80)

debug >>> get cr2
cr2 = 262144 (0x40000)

# Read bytes from large registers (offset, count)
debug >>> get r0 0 32          # 32 bytes from offset 0
debug >>> get acc 64 16        # 16 bytes from offset 64
debug >>> get rcyclic 128 64   # 64 bytes from offset 128

# Read as 32-bit words
debug >>> getw acc 0 8         # First 8 words of accumulator
debug >>> getw rcyclic 32 4    # 4 words from word offset 32
```

### Reading XMEM

Use `xmem` to inspect external memory by assembly row number or raw byte
address:

```text
debug >>> xmem row cr4 lr0 16 f32
debug >>> xmem row 4 1 128 u32
debug >>> xmem byte 0x1000 0 32 hex
debug >>> xmem byte 0x1000 0 32 int8
debug >>> xmem byte 0x1000 0 32 u8
debug >>> xmem byte 0x1000 0 16 cell16
debug >>> xmem byte cr3 lr2 16 f32
```

The syntax is:

```text
xmem row|byte BASE OFFSET COUNT hex|int8|u8|cell16|u32|f32
```

`BASE` and `OFFSET` may each be a decimal or hexadecimal immediate, an LR
register, or a CR register. The debugger adds the resolved values. In `row`
mode the sum is an assembly XMEM row number; in `byte` mode it is a raw byte
address.

The active execution mode determines the row size: 128 bytes normally and 512
bytes in wide-vector mode. Raw byte addressing can inspect the complete 8 MB
physical allocation.

`COUNT` is measured in bytes for `hex`, `int8`, and `u8`; in 16-bit values for
`cell16`; and in 32-bit values for `u32` and `f32`. `u8` values use three
decimal digits and `u32` values use ten decimal digits, both padded with leading
zeroes. Numeric formats use fixed-width display columns. `cell16` requires a
2-byte-aligned address, while the 32-bit formats require a 4-byte-aligned
address.

In wide-vector mode, `hex` output inserts two spaces after every group of four
bytes. Normal-mode hex output keeps one space between every byte.

`cell16` interprets each little-endian value as an ANSI terminal cell:

```text
bits  0..7  character index in the 256-character visible alphabet
bits 8..11  foreground color (0..15)
bits 12..15 background color (0..15)
```

The alphabet begins with visible ASCII and visible Latin-1 characters. The
remaining entries use Latin Extended-A characters so every byte value,
including index 200, maps to a printable character. The display inserts one
plain space after every two rendered `cell16` characters and prints 32
characters per output line. A complete 512-byte XMEM row therefore occupies
eight character lines. In the TUI, cursor and changed-value emphasis preserve
the encoded foreground and background instead of replacing the cell's color
pair. The XMEM title also reports the foreground and background values found
in the displayed range. For example, `fg=0/f bg=0/f` means that the memory
itself encodes only black and white; it does not indicate a terminal color
failure.

The display never combines values from two XMEM rows on one line. Each section
prints the XMEM row number and the real byte address where that row begins. If
a raw byte-addressed read starts in the middle of a row, the section also
prints the exact byte address where the requested display begins.

### Modifying State

```bash
debug >>> set lr0 100
Set lr0 = 100

debug >>> set cr5 0x8000
Set cr5 = 32768

debug >>> set pc 10
Set pc = 10
```

### Disassembly

```bash
debug >>> disasm
PC 3: break lr0 0; add lr0 lr0 0; mult_nop; acc_nop; b lr0 lr0 @4;;
```

### Saving State

```bash
debug >>> save my_debug_state.json
Registers saved to my_debug_state.json
```

Plain `save` remains register-only. To save XMEM with the registers, request a
selected row or byte range, or the complete physical allocation:

```text
debug >>> save row-state.json xmem row cr4 lr0 2
Registers saved to row-state.json
XMEM saved to row-state.xmem.bin

debug >>> save byte-state.json xmem byte 0x1000 0 512
debug >>> save full-state.json xmem all
debug >>> save "state with spaces.json" xmem all
```

For a row range, the final argument is the number of XMEM rows. For a byte
range, it is the number of bytes. The JSON file contains the registers plus
the resolved XMEM byte address, size, addressing mode, and sidecar filename.
The `.xmem.bin` sidecar contains the unmodified memory bytes. `xmem all` writes
the complete 8 MB allocation.
Quote a filename containing spaces when it is followed by `xmem` arguments.

The JSON file contains all register values:
```json
{
  "pc": 3,
  "lr": [0, 1280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "cr": [0, 131072, 262144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "r_regs": [[...], [...]],
  "r_cyclic": [...],
  "r_mask": [...],
  "acc": [...]
}
```

## Enabling Debug in Your Application

To add debug support to your IPU application, use `run_with_debug` with a debug callback:

```python
from ipu_emu.ipu_state import IpuState
from ipu_emu.emulator import run_with_debug
from ipu_emu.debug_cli import debug_prompt

state = IpuState()
state.load_program("program.bin")
# ... set up registers and memory ...

# Run with debug CLI (level 0-2 controls verbosity)
run_with_debug(state, lambda s, c: debug_prompt(s, c, level=1))
```

### Debug Levels

Pass the `level` parameter to `debug_prompt`:

```python
# Level 0: LR registers only
debug_prompt(state, cycle, level=0)

# Level 1: Also disassemble current instruction (default)
debug_prompt(state, cycle, level=1)

# Level 2: Auto-save registers to JSON
debug_prompt(state, cycle, level=2)
```
```

## Example Debug Session

```
$ cd src/tools/ipu-emu-py
$ uv run python run_fc_debug.py --dtype INT8

Running fully_connected (INT8) with debug CLI

========================================
IPU Debug - Break at PC=3
========================================
=== Program Counter ===
  PC = 3
=== LR Registers ===
  lr 0 =          0 (0x00000000)
  lr 1 =       1280 (0x00000500)
  ...

=== Current Instruction ===
  BREAK.IFEQ lr0 0; MULT_NOP; ACC_NOP; B lr0 lr0 @4;;

debug >>> get lr1
lr1 = 1280 (0x500)

debug >>> step
Stepping one instruction...

========================================
IPU Debug - Break at PC=4
========================================
...

debug >>> set lr0 256
Set lr0 = 256

debug >>> continue
Continuing execution...

IPU execution finished after 12847 cycles
```

## Tips

1. **Use conditional breaks** to stop at specific iterations:
   ```asm
   break.ifeq lr5 10 ;;   # Break on 10th iteration
   ```

2. **Step through loops** to watch register changes:
   ```
   debug >>> step
   debug >>> get lr5
   debug >>> step
   debug >>> get lr5
   ```

3. **Save state at key points** for offline analysis:
   ```
   debug >>> save before_mult.json
   debug >>> continue
   ```

4. **Use `getw` for accumulator** since it stores 32-bit values:
   ```
   debug >>> getw acc 0 16   # First 16 accumulator words
   ```

5. **Remember**: Debug mode requires an interactive terminal. Run directly with `uv run` or `python`, not via `bazel test`.

### Pipeline stage operations

The Disassembly **stages** view type (`f` to cycle formats) shows individual operations ready in the current
cycle, grouped as **CTRL → MULT → ACC → AAQ → STORE**. CTRL lists its active
LR, branch, and break commands separately. The XMEM input sideband and
simulation-only accumulator store have separate rows. Stages with no command
show **IDLE**; completed commands are not shown as occupying a stage.

The header identifies the cycle and PC. The debugger pauses before execution,
so active commands are marked **READY**. The emulator completes all slots of
one issue in one cycle, using each handler's snapshot/live operand semantics;
it does not expose intermediate hardware stage occupancy or stalls.

Step with `F8` while keeping this view open. Commands update from the actual
paused PC, including after branches and debugger PC edits. Use the wheel,
arrows, Page Up/Down, or Home/End to scroll. Commands wrap in narrow windows.
Press Escape in Disassembly to return to compact view. Each disassembly
tab retains its own view type. Other panes remain visible and interactive.
Help and execution controls remain available.

Disassembly and stage operations use semantic operand order from the
instruction specification, omitting unused encoding fields.
Disassembly highlights mnemonics and displays each operation's slot in a
right-hand column when space permits. Narrow panes reserve that space for
operands. Truncated operations end with an ellipsis; widen the pane to see more
of the instruction. The PC, breakpoint gutter, and parallel
operation grouping retain their keyboard and mouse behavior.

### Editing pipeline and memory values

Select a value in Pipeline Registers or XMEM and press `e`, click **e:Value**,
or double-click the value. The editor uses the active tab format: floating-point
for `f32`, signed integers for `int8`/`int32`, and unsigned integers for other
formats. Integer input accepts decimal, `0x` hex, and `0b` binary. `cell16` edits
the complete 16-bit cell. Enter applies; Escape cancels. Invalid or out-of-range
input leaves the bytes unchanged and keeps the editor open.

Edits affect only the selected value’s byte span. Pipeline edits use the displayed
register’s native or wide backing; XMEM edits use the resolved selected address.
Other memory tabs refresh immediately after applying an edit.

Escape sequence decoding waits at most 25 ms, so a standalone Escape closes
popups promptly while arrow/function-key sequences remain enabled.

Shift-Left moves the focused row’s divider left; Shift-Right moves it right,
regardless of which side of the divider has focus.

Help has an inset scrollbar: drag its thumb, click the track to page, or use
the mouse wheel. Its `[x]` button closes on mouse press or synthesized click;
release-only terminal events are also accepted.
