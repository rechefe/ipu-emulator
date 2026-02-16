# Debugging IPU Programs

The IPU emulator includes a powerful interactive debugger that allows you to pause execution, inspect registers, step through instructions, and modify state at runtime.

## Quick Start

### 1. Add Break Instructions to Your Assembly

Add `break` instructions where you want execution to pause:

```asm
main_loop:
    break ;;              # Unconditional break
    reset_acc;;
    ldr_cyclic_mult_reg lr0 cr0 lr15;;
```

### 2. Enable Debug Mode

Use `run_with_debug` or pass a debug callback when running the emulator:

```python
from ipu_emu.emulator import run_with_debug
from ipu_emu.debug_cli import debug_prompt

# Run with interactive debug CLI
run_with_debug(state, lambda s, c: debug_prompt(s, c, level=1))
```

Or use the fully_connected debug runner:

```bash
cd src/tools/ipu-emu-py
uv run python run_fc_debug.py --dtype INT8
```

### 3. Interactive Terminal Required

Debug mode requires interactive terminal access — run directly, not via `bazel test`.

When the break triggers, you'll enter an interactive debug prompt:

```
========================================
IPU Debug - Break at PC=3
========================================
=== Program Counter ===
  PC = 3
=== LR Registers ===
  lr 0 =          0 (0x00000000)
  lr 1 =       1280 (0x00000500)
  ...

debug >>> 
```

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

Control verbosity with `--debug-level=N`:

| Level | Description |
|-------|-------------|
| 0 | Print LR registers only |
| 1 | Also print disassembled current instruction (default) |
| 2 | Automatically save registers to JSON file |

Example:
```bash
bazel run //app -- <args> --debug --debug-level=2
```

## Debug Commands

### Navigation

| Command | Description |
|---------|-------------|
| `continue` / `c` | Continue execution until next break |
| `step` | Execute one instruction, then break again |
| `quit` / `q` | Halt execution and exit |

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
PC 3: break lr0 0; incr lr0 cr0 cr0 0; mult_nop; acc_nop; b lr0 lr0 @4;;
```

### Saving State

```bash
debug >>> save my_debug_state.json
Registers saved to my_debug_state.json
```

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
  break lr0 0; reset_acc; mult_nop; acc_nop; b lr0 lr0 @4;;

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
