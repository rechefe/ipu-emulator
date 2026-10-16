# IPU Emulator Documentation

[![CI](https://github.com/rechefe/ipu-emulator/actions/workflows/ci.yml/badge.svg)](https://github.com/rechefe/ipu-emulator/actions/workflows/ci.yml)
[![Documentation](https://github.com/rechefe/ipu-emulator/actions/workflows/docs.yml/badge.svg)](https://github.com/rechefe/ipu-emulator/actions/workflows/docs.yml)

The IPU (Inference Processing Unit) is a 128-lane VLIW vector processor for neural-network inference. This repository defines its instruction set, assembles programs for it, emulates it in Python, and holds the kernels that run on it. Bazel builds and tests everything.

| Section | Go here to |
|---|---|
| [Getting started](getting-started/index.md) | Install the toolchain, run the tests and run a first kernel. |
| [How-to](how-to/index.md) | Write, register or debug a kernel, or add an instruction. |
| [Reference](reference/index.md) | Look up an instruction, an operand type, a command, a kernel or a term. |
| [Explanation](explanation/index.md) | Understand the architecture, the timing model and the element formats. |
| [Specs and requirements](requirements/index.md) | Read the hardware block specifications and requirements. |
| [Open questions](requirements/open-questions.md) | See every open hardware, emulator and ISA question. |
| [Release log](release-log.md) | See what changed in each merged pull request. |

The source is on [GitHub](https://github.com/rechefe/ipu-emulator). [README.md](https://github.com/rechefe/ipu-emulator/blob/master/README.md) has the Quick Start.
