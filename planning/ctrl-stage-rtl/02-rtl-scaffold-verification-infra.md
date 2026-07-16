# RTL project scaffold + emulator-parity verification infrastructure

Part of the [CTRL-stage RTL epic](00-epic.md). Depends on #1.

## Goal

Stand up the RTL source layout, simulator toolchain, and — the part that
makes every later issue meaningful — a **parity-test harness** that can feed
identical stimulus into a cocotb testbench (RTL) and a live `ipu_emu.Ipu`
instance (Python), and diff the results. No RTL instruction handler in this
epic is considered done without a test built on this harness.

## Language and toolchain

- **RTL in SystemVerilog.** `stage-control.md`'s black-box diagram (§3.0)
  already uses SystemVerilog port syntax (`input logic`,
  `output logic [W-1:0]`) — follow that.
- **Simulator: Verilator**, driven by **cocotb** testbenches in Python. This
  is the natural fit here specifically because the golden model is already
  Python (`ipu_emu`) — a cocotb test can `import ipu_emu` directly and drive
  the DUT and the emulator from the same test file with no serialization
  boundary in between.
- **Bazel integration:** evaluate `rules_hdl` (Bazel rules for
  Verilog/Verilator/cocotb) as a `bazel_dep`, matching the project's existing
  practice of pulling in a rules_* module per new language
  (`rules_python`, and `rules_rust` proposed in the host-integration epic).
  If `rules_hdl`'s cocotb support proves too rough, a hermetic
  `genrule`/custom `use_repo_rule` wrapping `verilator` + `pytest`-invoked
  cocotb (same pattern already used for `generate_requirements` /
  `parse_pyproject_deps` in `MODULE.bazel`) is the fallback — either way, the
  simulator run must be reachable via `bazel test //...`, not a manual
  script.

## Proposed layout

```
src/hw/ipu-ctrl-rtl/
├── BUILD.bazel
├── rtl/
│   ├── ctrl_lr_alu.sv          # one LR-slot ALU lane (see #3)
│   ├── ctrl_lr_regfile.sv
│   ├── ctrl_cr_regfile.sv      # see #4
│   ├── ctrl_fetch.sv           # PC, inst mem, inst $ (see #4)
│   ├── ctrl_cond.sv            # branch/cond resolver (see #4)
│   ├── ctrl_dispatch.sv        # bus packing (see #5)
│   ├── ctrl_apb_slave.sv       # see #6
│   └── ctrl_stage.sv           # top-level integration (see #7)
└── test/
    ├── parity/                  # shared parity-harness helpers
    │   ├── golden.py            # thin wrapper: build an Ipu, feed one
    │   │                        # decoded VLIW word, read back state
    │   └── vectors.py           # shared instruction-operand generators
    ├── test_lr_alu.py           # cocotb, per #3
    ├── test_fetch_cond.py       # cocotb, per #4
    ├── test_dispatch.py         # cocotb, per #5
    └── test_apb_slave.py        # cocotb, per #6
```

## The parity harness (`test/parity/golden.py`)

The key design point: reuse the emulator's own decode/dispatch path as the
oracle, not a hand-rewritten reference model that could itself drift from
`ipu.py`. Concretely:

- Build instruction words with the existing assembler
  (`ipu_as.compound_inst` / `bazel run //src/tools/ipu-as-py:ipu-as`) so RTL
  tests exercise the *real* encoding via `SLOT_BINARY_LAYOUT`, never a
  hand-rolled bit-packing that could disagree with `instruction_spec.py`.
- Decode with `ipu_emu.execute.decode_instruction_word` — same function the
  real emulator uses.
- Drive one `Ipu` instance per test with `execute_vliw_cycle()` /
  `execute_vliw_cycle_skip_break()` and read back the exact state the
  cocotb test should assert against (LR file, PC, dispatch-bus fields,
  `xmem_read_addr`).
- Provide a small set of assertion helpers (e.g. `assert_lr_matches(dut,
  golden_state)`, `assert_dispatch_bus_matches(dut, bus_name,
  golden_bus_fields)`) so #3–#6's tests read as "run this instruction on
  both, assert equal," not ad hoc field pokes.

## Tests

- [ ] A trivial DUT (e.g. a pass-through register) builds and runs under
      `bazel test`, proving the toolchain is wired end-to-end.
- [ ] `golden.py` can decode an assembled instruction and report the exact
      LR/PC/bus values `ipu.py` would produce for it, with a test asserting
      that against a hand-computed expected value (i.e. the harness itself
      is tested once, directly, before anything is built on top of it).

## Acceptance Criteria

- [ ] `bazel build //src/hw/ipu-ctrl-rtl/...` and `bazel test
      //src/hw/ipu-ctrl-rtl/...` both work in the existing hermetic build.
- [ ] The parity harness decodes real assembled instructions and exposes
      emulator ground truth in a form cocotb tests can assert against
      directly.
- [ ] Directory/BUILD layout is in place for #3–#7 to add files to without
      further scaffolding work.
