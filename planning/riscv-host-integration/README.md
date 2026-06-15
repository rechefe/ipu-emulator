# RISC-V Host Integration — Issue Drafts

This directory holds ready-to-file GitHub issue drafts for the **RISC-V host
integration** feature. The accompanying design spec lives at
[`docs/content/specs/riscv-host-integration.md`](../../docs/content/specs/riscv-host-integration.md).

These are markdown drafts (the Cloud Agent cannot open GitHub issues directly);
paste each file's body into a new issue, or file them with `gh issue create`.

## Index

| # | File | Title | Labels |
|---|------|-------|--------|
| 0 | [`00-epic.md`](00-epic.md) | Epic: Emulated RISC-V host drives the IPU via a SystemRDL control register block | enhancement |
| 1 | [`01-systemrdl-regfile.md`](01-systemrdl-regfile.md) | Define the IPU host-control register block in SystemRDL + codegen | enhancement |
| 2 | [`02-emulator-mmio-model.md`](02-emulator-mmio-model.md) | Emulator MMIO control-register model mapped onto `IpuState` | enhancement |
| 3 | [`03-steppable-engine.md`](03-steppable-engine.md) | Refactor the run loop into a host-drivable, steppable execution engine | enhancement |
| 4 | [`04-unicorn-riscv.md`](04-unicorn-riscv.md) | Integrate an emulated RISC-V core (Unicorn) with the IPU MMIO bridge | enhancement |
| 5 | [`05-rust-firmware.md`](05-rust-firmware.md) | Rust `no_std` firmware + generated driver for the IPU host | enhancement |
| 6 | [`06-end-to-end.md`](06-end-to-end.md) | End-to-end firmware-driven app run, parity tests, and docs | enhancement |

## Suggested order

```
1 ──▶ 2 ──▶ 4 ──▶ 5 ──▶ 6
  └──▶ 3 ──┘
```

Issue 1 fixes the register contract. Issues 2 and 3 can proceed in parallel,
then 4 wires the host in, 5 adds firmware, and 6 ties it all together.
