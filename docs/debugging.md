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

### 2. Enable Debug Mode in Your Host Code

In your C application, set `debug_config.enabled = true` in the emulator config:

```c
emulator__test_config_t config = {
    .test_name = "My IPU Application",
    .max_cycles = 1000000,
    .progress_interval = 100,
    .setup = my_setup,
    .teardown = my_teardown,
    .debug_config = {
        .enabled = true,           // Enable debug mode
        .level = IPU_DEBUG_LEVEL_1 // Show disassembly
    }
};

return emulator__run_test(argc, argv, &config);
```

### 3. Run with `bazel run` (Not `bazel test`)

Debug mode requires interactive terminal access:

```bash
bazel run //src/apps/myapp:myapp -- <program.bin> [args...]
```

!!! warning
    `bazel test` runs in a sandbox without terminal access. Always use `bazel run` for interactive debugging.

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

To add debug support to your IPU application, configure the `debug_config` in your emulator test config:

```c
#include "emulator/emulator.h"

int main(int argc, char **argv) {
    emulator__test_config_t config = {
        .test_name = "My IPU Application",
        .max_cycles = 1000000,
        .progress_interval = 100,
        .setup = my_setup,
        .teardown = my_teardown,
        .debug_config = {
            .enabled = true,            // Set to true to enable
            .level = IPU_DEBUG_LEVEL_1  // Debug verbosity level
        }
    };

    return emulator__run_test(argc, argv, &config);
}
```

### Optional: Command-Line Flag

You can add a command-line flag to toggle debug mode at runtime:

```c
#include "emulator/emulator.h"
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    // Parse --debug flag
    bool debug_enabled = false;
    int debug_level = 1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0) {
            debug_enabled = true;
        } else if (strncmp(argv[i], "--debug-level=", 14) == 0) {
            debug_level = atoi(argv[i] + 14);
            debug_enabled = true;
        }
    }

    emulator__test_config_t config = {
        .test_name = "My IPU Application",
        .max_cycles = 1000000,
        .progress_interval = 100,
        .setup = my_setup,
        .teardown = my_teardown,
        .debug_config = {
            .enabled = debug_enabled,
            .level = (ipu_debug__level_t)debug_level
        }
    };

    return emulator__run_test(argc, argv, &config);
}
```

Then run with:
```bash
bazel run //src/apps/myapp:myapp -- program.bin --debug
bazel run //src/apps/myapp:myapp -- program.bin --debug-level=2
```
```

## Example Debug Session

```
$ bazel run //src/apps/myapp:myapp -- program.bin data.bin

INFO: My IPU Application Started
INFO: Starting IPU execution with debug mode...
INFO: Break triggered at PC=3, entering debug prompt...

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

INFO: IPU execution finished after 12847 cycles
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

5. **Remember**: `bazel test` runs in a sandbox without terminal access. Use `bazel run` for interactive debugging.
