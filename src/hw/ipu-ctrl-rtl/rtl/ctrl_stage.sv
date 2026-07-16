// ctrl_stage.sv — CTRL-stage top level. Port list transcribed from the
// black-box diagram in docs/content/specs/stage-control.md section 3.0.
//
// STATUS: scaffold. Only the LR-slot register file + ALU (ctrl_lr_stage,
// planning/ctrl-stage-rtl/03-lr-regfile-alu.md) is real. Every other block
// is an explicit TODO citing the planning issue that owns it — see
// src/hw/ipu-ctrl-rtl/rtl/README.md for the current status of each one.
// Bus widths marked "impl" in the spec (MULT_BUS_W and friends) are left
// as placeholder 1-bit ties; do not wire anything to them yet.
//
// This module has never been run through a simulator (none is installed
// in this environment) — see the README for what "verified" will mean
// once planning issue #2 lands the toolchain.

module ctrl_stage
  import ctrl_pkg::*;
#(
    // Placeholder widths for the four dispatch buses -- "impl" in
    // stage-control.md section 4 until issue #5 defines the real packed
    // layout (slot fields + resolved CR/LR operand values per bus).
    parameter int unsigned MULT_BUS_W = 1,
    parameter int unsigned ACC_BUS_W  = 1,
    parameter int unsigned AAQ_BUS_W  = 1,
    parameter int unsigned STR_BUS_W  = 1,
    parameter int unsigned XMEM_ADDR_W = 16,
    parameter int unsigned APB_ADDR_W = 32,
    parameter int unsigned APB_DATA_W = 32
) (
    input  logic clk,
    input  logic rst,

    // APB slave -- host configuration (issue #6).
    input  logic                     apb_psel,
    input  logic                     apb_penable,
    input  logic                     apb_pwrite,
    input  logic [APB_ADDR_W-1:0]    apb_paddr,
    input  logic [APB_DATA_W-1:0]    apb_pwdata,
    output logic [APB_DATA_W-1:0]    apb_prdata,
    output logic                     apb_pready,
    output logic                     apb_pslverr,

    // Dispatch buses into the MULT stage (issue #5).
    output logic [MULT_BUS_W-1:0]    mult_vliw_bus,
    output logic [ACC_BUS_W-1:0]     acc_vliw_bus,
    output logic [AAQ_BUS_W-1:0]     aaq_vliw_bus,
    output logic [STR_BUS_W-1:0]     str_vliw_bus,

    // XMEM read-address path (issue #5).
    output logic [XMEM_ADDR_W-1:0]   xmem_read_addr,
    output logic                     xmem_read_en

    // NOT YET IN THE SPEC -- planning/ctrl-stage-rtl/01-spec-emulator-reconciliation.md
    // gap #1: the break slot (BREAK/BREAK.IFEQ) and the cond-slot's BKPT
    // opcode have no port in stage-control.md section 3 at all, despite
    // being real, hardware-classified instructions
    // (SLOT_METADATA["break"] is not marked hardware:False) that gate
    // every other slot's side effects for the cycle
    // (Ipu.execute_vliw_cycle dispatches "break" first and returns early
    // on BreakResult.BREAK -- ipu.py:1237). Once issue #1 defines the
    // real port (e.g. `halted_o` / `break_hit_o`), add it here -- do not
    // silently drop this behavior from the RTL just because the spec
    // doesn't have a port for it yet.
);

  // -----------------------------------------------------------------------
  // LR-slot register file + 3-lane ALU — REAL, see
  // planning/ctrl-stage-rtl/03-lr-regfile-alu.md and rtl/ctrl_lr_stage.sv.
  // `lr_inst` below is a placeholder source for the 3 raw LR sub-
  // instructions: real hardware demuxes them out of `inst $` (issue #4's
  // fetch path), which doesn't exist in this module yet, so they're tied
  // to NOP for now rather than left floating.
  // -----------------------------------------------------------------------
  logic [LR_INST_W-1:0] lr_inst [LR_LANES];
  always_comb begin
    for (int unsigned i = 0; i < LR_LANES; i++) begin
      lr_inst[i] = {LR_OP_NOP, {(LR_INST_W - LR_OPCODE_W){1'b0}}};
    end
  end

  // TODO(#4): CR register file. Tied to all-zero here so ctrl_lr_stage has
  // something to read; CR0/CR1's hard-wired 0/1 values are issue #4's
  // responsibility, not modeled here.
  logic [REG_WIDTH-1:0] cr_rdata [CR_COUNT];
  always_comb begin
    for (int unsigned i = 0; i < CR_COUNT; i++) begin
      cr_rdata[i] = '0;
    end
  end

  logic [REG_WIDTH-1:0] lr_rdata_postwrite [LR_COUNT];  // for issue #5's dispatch bus packing
  logic [REG_WIDTH-1:0] lr_rdata_snapshot  [LR_COUNT];  // for issue #4's cond-slot resolver
  logic                 lr_conflict;                    // diagnostic only, see ctrl_lr_regfile.sv

  ctrl_lr_stage u_lr_stage (
      .clk                 (clk),
      .rst                 (rst),
      .lr_inst_i           (lr_inst),
      .cr_rdata_i          (cr_rdata),
      .lr_rdata_postwrite_o(lr_rdata_postwrite),
      .lr_rdata_snapshot_o (lr_rdata_snapshot),
      .conflict_o          (lr_conflict)
  );

  // -----------------------------------------------------------------------
  // TODO(#4): PC, dual-bank inst mem, inst $, cond/branch resolver.
  // TODO(#5): dispatch-bus packing (mult/acc/aaq/str_vliw_bus), XMEM
  //           read-address resolution, BREAK/BKPT priority gating.
  // TODO(#6): APB slave (IMEM/CR inactive-bank writes, bank swap regs).
  //
  // Every output below is tied to a safe, inert default so this module
  // elaborates; none of it is real yet.
  // -----------------------------------------------------------------------
  assign apb_prdata     = '0;
  assign apb_pready     = 1'b0;
  assign apb_pslverr    = 1'b0;

  assign mult_vliw_bus  = '0;
  assign acc_vliw_bus   = '0;
  assign aaq_vliw_bus   = '0;
  assign str_vliw_bus   = '0;

  assign xmem_read_addr = '0;
  assign xmem_read_en   = 1'b0;

endmodule : ctrl_stage
