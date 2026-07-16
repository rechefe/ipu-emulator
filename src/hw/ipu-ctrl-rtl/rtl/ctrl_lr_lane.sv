// ctrl_lr_lane.sv — one LR-slot ALU lane: decode + execute for a single
// 17-bit LR sub-instruction. Three instances run in parallel per VLIW word
// (ctrl_pkg::LR_LANES = 3); cross-lane same-destination conflict checking
// and the actual register write live in ctrl_lr_regfile.sv, not here.
//
// This is a combinational block: it reads the LR/CR values visible at the
// *start* of the current cycle (the register file's current flop outputs)
// and produces the write-enable/address/data for that same cycle's write.
// This is deliberate and needs no extra "snapshot" machinery — unlike the
// Python emulator, which has to deep-copy the register file every cycle to
// fake read-before-write semantics (Ipu.execute_vliw_cycle's
// `self.snapshot = self.state.regfile.snapshot()`), a synchronous register
// file gets read-before-write for free: combinational reads see the old
// value; the write only lands on the next clock edge. `lr_rdata_i` /
// `cr_rdata_i` below correspond exactly to what the emulator calls "the
// snapshot."
//
// Ground truth for every opcode's arithmetic: the execute_lr_* handlers in
// src/tools/ipu-emu-py/src/ipu_emu/ipu.py (line numbers as of this
// writing; re-check if ipu.py has moved on since):
//   execute_lr_set            ipu.py:511   dest = CR[src]              (no immediate mode -- see ctrl_pkg.sv)
//   execute_lr_add            ipu.py:515   dest = src_a + src_b        (20-bit wrap)
//   execute_lr_sub            ipu.py:519   dest = src_a - src_b        (20-bit wrap)
//   execute_lr_incr_mod_pow2  ipu.py:535   dest = (dest + step) & ((1<<k_exp)-1)
//   execute_lr_inc            ipu.py:523   dest = dest + imm           (20-bit wrap, dest read-modify-write)
//   execute_lr_dec            ipu.py:529   dest = dest - imm           (20-bit wrap, dest read-modify-write)
//
// See planning/ctrl-stage-rtl/03-lr-regfile-alu.md.

module ctrl_lr_lane
  import ctrl_pkg::*;
(
    // Raw 17-bit LR sub-instruction for this lane (opcode MSBs, then the
    // three union fields, matching SLOT_BINARY_LAYOUT["lr"] field order).
    input  logic [LR_INST_W-1:0]      lr_inst_i,

    // Register read ports -- current (pre-this-cycle-write) values.
    input  logic [REG_WIDTH-1:0]      lr_rdata_i [LR_COUNT],
    input  logic [REG_WIDTH-1:0]      cr_rdata_i [CR_COUNT],

    // Resolved write-back for this lane, consumed by ctrl_lr_regfile.sv.
    output logic                      lr_we_o,
    output logic [LR_IDX_W-1:0]       lr_waddr_o,
    output logic [REG_WIDTH-1:0]      lr_wdata_o
);

  // ---------------------------------------------------------------------
  // Field demux -- mirrors SLOT_UNIONS["lr"].opcode_bindings: which raw
  // field each opcode's named operand comes from. This is the "slot
  // demuxing" stage-control.md section 5 says CTRL performs (IMEM entries
  // are pre-decoded; only demuxing happens in hardware, no opcode-level
  // decode of a raw instruction word).
  // ---------------------------------------------------------------------
  logic [LR_OPCODE_W-1:0]        opcode_raw;
  lr_opcode_e                    opcode;
  logic [LR_FIELD0_W-1:0]        field0;   // SET.src(CrIdx) | ADD/SUB.src_b, INCR_MOD_POW2.step (LcrIdx) | INC/DEC.imm
  logic [LR_FIELD1_W-1:0]        field1;   // dest / reg (LrIdx)
  logic [LR_FIELD2_W-1:0]        field2;   // ADD/SUB.src_a (LrIdx) | INCR_MOD_POW2.k

  assign {opcode_raw, field0, field1, field2} = lr_inst_i;
  assign opcode = lr_opcode_e'(opcode_raw);

  // ---------------------------------------------------------------------
  // LcrIdx resolution -- ipu.py Ipu._resolve_operand():
  //   if raw_value < LR_REG_COUNT: get_lr(raw_value)
  //   else:                        get_cr(raw_value - LR_REG_COUNT)
  // ---------------------------------------------------------------------
  function automatic logic [REG_WIDTH-1:0] resolve_lcr(
      input logic [LCR_IDX_W-1:0] raw_idx
  );
    if (raw_idx < LCR_IDX_W'(LR_COUNT))
      return lr_rdata_i[LR_IDX_W'(raw_idx)];
    else
      return cr_rdata_i[CR_IDX_W'(raw_idx - LCR_IDX_W'(LR_COUNT))];
  endfunction

  // Per-opcode operand values, computed unconditionally (cheap, combinational)
  // and muxed onto the output by opcode below -- keeps the write-back mux a
  // flat case statement instead of nested nesting per opcode.
  logic [LR_IDX_W-1:0]      dest_idx;
  logic [REG_WIDTH-1:0]     dest_snapshot;   // this lane's dest read *before* its own write
  logic [REG_WIDTH-1:0]     src_a_val;       // ADD/SUB.src_a (LrIdx, field2[LR_IDX_W-1:0])
  logic [REG_WIDTH-1:0]     src_b_or_step;   // ADD/SUB.src_b | INCR_MOD_POW2.step (LcrIdx, field0)
  logic [CR_IDX_W-1:0]      set_cr_idx;      // SET.src (CrIdx, field0[CR_IDX_W-1:0] -- field0[4] unused)
  logic [LR_MOD_POW2_K_FIELD_W-1:0] k_encoded; // INCR_MOD_POW2.k (field2[3:0]; field2[4] is the pad bit)
  logic [LR_INC_DEC_IMM_W-1:0] imm;          // INC/DEC.imm (field0, unsigned)

  assign dest_idx      = field1;
  assign dest_snapshot = lr_rdata_i[dest_idx];
  assign src_a_val     = lr_rdata_i[field2[LR_IDX_W-1:0]];
  assign src_b_or_step = resolve_lcr(field0);
  assign set_cr_idx    = field0[CR_IDX_W-1:0];
  assign k_encoded     = field2[LR_MOD_POW2_K_FIELD_W-1:0];
  assign imm           = field0;

  // INCR_MOD_POW2 mask -- k_exp = k_encoded + LR_MOD_POW2_K_MIN (range [1,9]).
  // Computed at REG_WIDTH so the shift can't overflow the mask width.
  logic [3:0] k_exp;
  logic [REG_WIDTH-1:0] mod_pow2_mask;
  assign k_exp         = k_encoded + 4'(LR_MOD_POW2_K_MIN);
  assign mod_pow2_mask = (REG_WIDTH'(1) << k_exp) - REG_WIDTH'(1);

  // ---------------------------------------------------------------------
  // Write-back mux
  // ---------------------------------------------------------------------
  always_comb begin
    lr_we_o    = 1'b1;
    lr_waddr_o = dest_idx;
    lr_wdata_o = '0;

    unique case (opcode)
      LR_OP_SET: lr_wdata_o = cr_rdata_i[set_cr_idx];

      LR_OP_ADD: lr_wdata_o = src_a_val + src_b_or_step;   // 20-bit wrap by construction
      LR_OP_SUB: lr_wdata_o = src_a_val - src_b_or_step;   // 20-bit wrap by construction

      LR_OP_INCR_MOD_POW2:
        lr_wdata_o = (dest_snapshot + src_b_or_step) & mod_pow2_mask;

      LR_OP_INC: lr_wdata_o = dest_snapshot + REG_WIDTH'(imm);
      LR_OP_DEC: lr_wdata_o = dest_snapshot - REG_WIDTH'(imm);

      default: begin // LR_OP_NOP and any reserved encoding
        lr_we_o    = 1'b0;
        lr_waddr_o = '0;
        lr_wdata_o = '0;
      end
    endcase
  end

endmodule : ctrl_lr_lane
