# IPU Emulator Documentation

Welcome to the IPU Emulator documentation. The IPU (Inference Processing Unit) is a 128-lane VLIW vector processor for neural-network inference. This repository defines its instruction set, assembles programs for it, emulates it, and holds the kernels that run on it.

## Overview

This repository contains:

- **ISA definition** (`ipu_common`) - `instruction_spec.py`, the single source of truth for instructions, operands and registers
- **IPU assembler** (`ipu_as`) - Python assembler with Jinja2 preprocessing, plus the generated SystemVerilog package
- **IPU emulator** (`ipu_emu`) - Python emulator of the IPU, with an interactive debugger
- **Kernels** (`ipu_apps`) - assembly kernels (matmul, convolutions, softmax, normalize and others), each with its harness and tests
- **Build system** - Bazel builds, tests and generates these docs

## Quick Links

- [Getting Started guide (PDF)](https://github.com/rechefe/ipu-emulator/releases/download/guide-latest/getting-started.pdf) — rendered from `docs/guide` on every change to master
- [Assembly Syntax Guide](assembly-syntax.md) — generated (register tokens from `ipu_common`)
- [Operand types](operand-types.md) — generated reference for `instruction_spec` field types
- [Instruction Reference](instructions.md) — generated per-opcode pages
- [Repository on GitHub](https://github.com/rechefe/ipu-emulator)

---

## Repository Structure

```
ipu-emulator/
├── src/tools/
│   ├── ipu-common/         # ISA definition (ipu_common)
│   ├── ipu-as-py/          # Assembler (ipu_as)
│   ├── ipu-emu-py/         # Emulator and debugger (ipu_emu)
│   └── ipu-apps/           # Kernels and their harnesses (ipu_apps)
├── docs/                   # MkDocs config, hooks, overrides
│   └── content/            # Documentation page sources (MkDocs docs_dir)
├── BUILD.bazel             # Root build configuration
├── MODULE.bazel            # Bazel module dependencies
└── README.md               # Project README
```

---

## Getting Started

### Assembly Language

The IPU supports a custom assembly language with powerful preprocessing capabilities. See the [Assembly Syntax Guide](assembly-syntax.md) for:

- [Instruction format](instruction-format.md) — generated bit layout, C header and SystemVerilog package downloads
- Basic instruction format and syntax
- Instruction separators (`;` vs `;;`)
- Register types and immediate values  
- Labels and control flow
- Jinja2 preprocessing (variables, loops, macros, conditionals)
- Complete examples and usage guide

### Building and Testing

**Build everything:**

```bash
bazel build //...
```

**Run tests:**

```bash
bazel test //...
```

**Generate documentation:**

```bash
bazel build //docs:build_docs
```

## Next Steps

- [Building Applications](building-applications.md) - Learn how to build IPU applications
- [IPU Configuration](ipu-configuration.md) - Configure dtype, CR constants, and dstructure element counts
- [Assembly Syntax Guide](assembly-syntax.md) — syntax, compounds, Jinja2 (generated)
- [Operand types](operand-types.md) — type names used in the instruction reference tables
- [Instruction Reference](instructions.md) — all opcodes (generated)

## Contributing

See [README.md](https://github.com/rechefe/ipu-emulator/blob/master/README.md) for development setup and contribution guidelines.

## Additional Resources

- [Jinja2 Template Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)
- [Bazel Build System](https://bazel.build/)
- [GitHub Repository](https://github.com/rechefe/ipu-emulator)
