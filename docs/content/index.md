# IPU Emulator Documentation

Welcome to the IPU Emulator documentation. This project provides a complete IPU (Intelligence Processing Unit) emulation environment with assembly language support, C-based emulator, and comprehensive tooling.

## Overview

This repository contains:

- **IPU Emulator** - C-based emulator for IPU behavior
- **IPU Assembly Language** (`ipu-as`) - Python-based assembler with Jinja2 preprocessing
- **XMEM Support** - External memory emulation
- **Sample Programs** - Fully connected neural network examples
- **Build System** - Bazel-based build and test infrastructure

## Quick Links

- [Assembly Syntax Guide](assembly-syntax.md) — generated (register tokens from `ipu_common`)
- [Operand types](operand-types.md) — generated reference for `instruction_spec` field types
- [Instruction Reference](instructions.md) — generated per-opcode pages
- [Repository on GitHub](https://github.com/rechefe/ipu-emulator)

---

## Repository Structure

```
ipu-c-samples/
├── src/
│   ├── apps/               # Sample applications
│   ├── lib/                # Core libraries (ipu, xmem, emulator, fp, logging)
│   └── tools/              # IPU assembler and emulator tools
├── test/                   # Unit tests
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
- [IPU Configuration](ipu-configuration.md) - Configure dtype, CR constants, and dstructure lane counts
- [Assembly Syntax Guide](assembly-syntax.md) — syntax, compounds, Jinja2 (generated)
- [Operand types](operand-types.md) — type names used in the instruction reference tables
- [Instruction Reference](instructions.md) — all opcodes (generated)

## Contributing

See [README.md](https://github.com/rechefe/ipu-emulator/blob/main/README.md) for development setup and contribution guidelines.

## Additional Resources

- [Jinja2 Template Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)
- [Bazel Build System](https://bazel.build/)
- [GitHub Repository](https://github.com/rechefe/ipu-emulator)
