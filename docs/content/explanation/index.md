# Explanation

These pages are for understanding how the IPU and its tools work and why they are built this way.

| Page | What it covers |
|---|---|
| [Architecture](architecture.md) | The packages, how they depend on each other, how a program goes from source to a checked run, and prototyping mode. |
| [Timing model](timing-model.md) | Which register values each stage reads within a cycle. |
| [Element formats](element-formats.md) | Each element format, how it encodes values, how a CR scalar is decoded in it, and how a kernel selects one. |
| [ISA idioms](isa-idioms.md) | The instruction sequences kernels use for operations the ISA lacks, why, and what each costs. |
| [Kernel coverage](kernel-coverage.md) | How a computation resolves to a kernel, and which operations the kernels cover. |

The [architecture decision records](https://github.com/rechefe/ipu-emulator/tree/master/docs/adr) in `docs/adr/` record the decisions behind the design.
