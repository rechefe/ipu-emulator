# Integrate an emulated RISC-V core (Unicorn) with the IPU MMIO bridge

Part of the [RISC-V host integration epic](00-epic.md). Depends on #1, #2, #3.

## Goal

Embed an emulated **RISC-V** core in the Python environment using
[Unicorn](https://www.unicorn-engine.org/), expose the IPU host-control register
block (#1/#2) as an **MMIO** window, and bridge Unicorn's memory callbacks into
the control-register model so firmware running on the core can drive the IPU.

## Implementation

- Add the `unicorn` Python package to the relevant `pyproject.toml`; regenerate
  the requirements lock; confirm the RISC-V backend is available
  (`UC_ARCH_RISCV`, `UC_MODE_RISCV32`/`RISCV64`).
- Add a host module (e.g. `ipu_emu/riscv_host.py`) that:
  - Creates a `Uc` instance, maps a **RAM** region for firmware code/data/stack,
    and `mmio_map`s the IPU control window.
  - Routes MMIO `read`/`write` callbacks to `IpuHostController` (#2).
  - Loads a flat firmware binary, sets the reset PC, and runs via `emu_start`.
- Define the address map (RAM base/size, MMIO base/size) and keep it consistent
  with the firmware linker layout (#5).
- **Execution model:** run the IPU functionally inside the `START`/`STEP` write
  callback (deterministic, no threads). Document the alternative cooperative
  model and why it was deferred (see spec §10).

```python
uc = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32)
uc.mem_map(RAM_BASE, RAM_SIZE)
uc.mem_write(RAM_BASE, firmware_image)
uc.mmio_map(MMIO_BASE, MMIO_SIZE, read_cb=bridge.on_read, write_cb=bridge.on_write)
uc.reg_write(UC_RISCV_REG_PC, RAM_BASE)
uc.emu_start(RAM_BASE, RAM_BASE + len(firmware_image))
```

## Docs

- Document the Unicorn integration and address map in the spec.

## Tests

- [ ] A minimal RISC-V test stub (hand-written words or a tiny assembled blob)
      that does `sw` to `STATUS`/`CR` offsets triggers the expected control-model
      side effects through the MMIO bridge.
- [ ] Reads of `STATUS`/`PC`/`ID` return the correct values to the core.
- [ ] `bazel test //...` passes with `unicorn` available.

## Acceptance Criteria

- [ ] Unicorn RISC-V core boots in-process and reaches the firmware entry.
- [ ] MMIO reads/writes are correctly bridged to the control-register model.
- [ ] The IPU runs to halt in response to a `CTRL.START` store from the core.
