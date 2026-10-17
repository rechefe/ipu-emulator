# Explanation

These pages are for understanding how the IPU and its tools work and why they are built this way.

| Page | What it covers |
|---|---|
| [Architecture](architecture.md) | The packages, how they depend on each other, and how one cycle runs. |
| [Prototyping mode](../prototyping-mode.md) | The opt-in fast paths and how they differ from the faithful model. |
| [Timing model](timing-model.md) | Which register values each stage reads within a cycle. |
| [Element formats](element-formats.md) | Each element format, how it encodes values, how a CR scalar is decoded in it, and how a kernel selects one. |
| [ISA idioms](../isa-alias-catalogue.md) | Recurring instruction patterns in the kernels, and what each costs. |
| [Kernel coverage](../app-coverage.md) | Which operations the kernels cover. |

The [architecture decision records](https://github.com/rechefe/ipu-emulator/tree/master/docs/adr) in `docs/adr/` record the decisions behind the design.
