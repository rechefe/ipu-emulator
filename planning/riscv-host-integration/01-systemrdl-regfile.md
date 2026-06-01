# Define the IPU host-control register block in SystemRDL + codegen

Part of the [RISC-V host integration epic](00-epic.md).

## Goal

Author the IPU **host-control / configuration register block** in **SystemRDL**
(`ipu_ctrl.rdl`) as the single source of truth, and wire the
[PeakRDL](https://peakrdl.readthedocs.io/) toolchain into Bazel to generate:

1. Python field offsets / masks / enums for the emulator control model (Issue #2).
2. A PAC-like **Rust** register-access crate (`ipu-pac`) for the firmware (Issue #5).
3. Register-map documentation (HTML/Markdown) linked from the docs site.

No consumer hand-duplicates register metadata.

## Register map (starting point)

Implements the layout proposed in
[`specs/riscv-host-integration.md` §4](../../docs/content/specs/riscv-host-integration.md).
Final offsets/fields are owned by the `.rdl` file.

- `ID` (RO) — magic + version
- `CTRL` (RW/W1S) — `START`, `HALT`, `STEP`, `RESET`, `IMEM_SWAP`, `CONTINUE`
- `STATUS` (RO) — `RUNNING`, `HALTED`, `BREAK`, `ERROR`, `ERR_CODE`
- `PC` (RW), `CYCLES` (RO), `MAX_CYCLES` (RW)
- `DTYPE` (RW), `DSTRUCTURE` (RW), `ELU_ALPHA` (RW)
- `CR[0..15]` (RW\*; `CR0`/`CR1` hard-wired, `CR15` aliases `DSTRUCTURE`)
- `IMEM_ADDR` (RW), `IMEM_WDATA` (WO), `IMEM_RDATA` (RO), `IMEM_CTRL` (RW)

## Implementation

- Add `src/hw/ipu_ctrl.rdl` (or `src/tools/ipu-ctrl-rdl/`) with the register block.
- Add a Bazel rule wrapping PeakRDL exporters so generation is hermetic:
  - Python: `peakrdl-python` **or** a small custom exporter over the
    `systemrdl-compiler` IR (decide via a short spike — see note below).
  - Rust: RDL→SVD→`svd2rust` **or** a custom Jinja Rust template over the same IR.
  - Docs: `peakrdl-html`.
- Add `peakrdl`, `peakrdl-html`, and any exporter packages to the relevant
  `pyproject.toml`; regenerate the requirements lock.

> **Spike (Rust path):** RDL→SVD→`svd2rust` is the most "standard" Rust route but
> the RDL→SVD exporter quality varies. Since `systemrdl-compiler` exposes a clean
> Python IR, a single Jinja template emitting *both* the Python constants and a
> `no_std` Rust module from that IR may be the least-fragile option. Pick one and
> document the choice.

## Docs

- Link the generated register-map page from the docs nav.
- Cross-reference [`specs/riscv-host-integration.md`](../../docs/content/specs/riscv-host-integration.md)
  and [`specs/stage-control.md`](../../docs/content/specs/stage-control.md).

## Tests

- [ ] `bazel build` generates Python, Rust, and docs artifacts from `ipu_ctrl.rdl`.
- [ ] A test asserts the generated Python field offsets/masks match the `.rdl`
      (e.g. `CTRL.START` bit position, `DSTRUCTURE` field layout).
- [ ] Generated Rust crate compiles for the bare-metal RISC-V target.

## Acceptance Criteria

- [ ] `ipu_ctrl.rdl` exists and is the single source for all register metadata.
- [ ] PeakRDL generation is wired into Bazel and hermetic.
- [ ] Python, Rust, and HTML artifacts all generate from the one source.
- [ ] Rust-codegen path chosen and documented.
- [ ] `bazel test //...` passes with the new dependencies.
