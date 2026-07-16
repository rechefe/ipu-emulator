# CTRL-Stage RTL — Issue Drafts

This directory holds ready-to-file GitHub issue drafts for developing
**synthesizable RTL for the CTRL stage** — the pipeline stage described in
[`docs/content/specs/stage-control.md`](../../docs/content/specs/stage-control.md).

The core discipline of this epic: the spec doc describes the stage's
external contract, but the **Python emulator**
(`src/tools/ipu-emu-py/src/ipu_emu/ipu.py` and friends) is the tested,
executable ground truth for exact instruction semantics, and the two have
already been found to disagree in a few places (see `00-epic.md`). No RTL
instruction handler is considered correct until it's checked against the
emulator, not just the prose — see `02-rtl-scaffold-verification-infra.md`
for the parity-testing approach that runs throughout.

These are markdown drafts (the Cloud Agent cannot open GitHub issues
directly); paste each file's body into a new issue, or file them with
`gh issue create`.

## Index

| # | File | Title |
|---|------|-------|
| 0 | [`00-epic.md`](00-epic.md) | Epic: CTRL-stage RTL, verified against the Python emulator as golden model |
| 1 | [`01-spec-emulator-reconciliation.md`](01-spec-emulator-reconciliation.md) | Reconcile `stage-control.md` against the emulator before any RTL is written |
| 2 | [`02-rtl-scaffold-verification-infra.md`](02-rtl-scaffold-verification-infra.md) | RTL project scaffold + emulator-parity verification infrastructure |
| 3 | [`03-lr-regfile-alu.md`](03-lr-regfile-alu.md) | LR register file + 3-lane LR ALU RTL, verified against `execute_lr_*` |
| 4 | [`04-cr-regfile-fetch-branch.md`](04-cr-regfile-fetch-branch.md) | CR register file, PC, dual-bank fetch path, and cond/branch resolver |
| 5 | [`05-dispatch-buses-break-priority.md`](05-dispatch-buses-break-priority.md) | Dispatch-bus packing, XMEM read-address path, and BREAK/BKPT priority gating |
| 6 | [`06-apb-host-config.md`](06-apb-host-config.md) | APB slave register block (host config + bank swap) |
| 7 | [`07-integration-parity-signoff.md`](07-integration-parity-signoff.md) | Full CTRL-stage integration, app-level parity suite, sign-off |

## Suggested order

```
1 ──▶ 2 ──▶ 3 ──┐
           4 ──┼──▶ 5 ──▶ 6 ──▶ 7
```

Issue 1 is a documentation/decision pass and blocks everything — the RTL
work in 3–7 depends on it having settled the break/BKPT, PC-width, and
IMEM-sizing questions first. Issue 2 stands up the toolchain and the
emulator-parity harness that 3 through 7 all build tests on. Issues 3 and 4
can proceed in parallel once 2 lands; 5 needs both; 6 depends only on 4 but
coordinates with the separate
[RISC-V host integration epic](../riscv-host-integration/00-epic.md)'s
register-map work rather than duplicating it; 7 integrates everything and
is the epic's sign-off gate.

## Relationship to the RISC-V host integration epic

[`planning/riscv-host-integration/`](../riscv-host-integration/) is a
sibling epic covering the **emulator's** host/APB/MMIO model (Unicorn RISC-V
core, SystemRDL register block, Rust firmware). This epic's Issue 6 (APB
slave RTL) is the natural RTL counterpart to that epic's register-map work
and should share its SystemRDL source rather than re-describing the same
bits independently — see `06-apb-host-config.md` for how the two are meant
to stay in sync without either blocking the other's start.
