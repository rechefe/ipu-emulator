// ctrl_lr_stage.sv — wires up the 3 LR-slot lanes and the LR register file.
// This is the full, self-contained deliverable for
// planning/ctrl-stage-rtl/03-lr-regfile-alu.md: given the current VLIW
// word's 3 LR sub-instructions and a CR read port (ctrl_cr_regfile.sv is
// issue #4's block; here it's just an input), it executes all three lanes
// and updates the LR file. It intentionally does *not* attempt to be
// wired into a full CTRL stage yet -- see ctrl_stage.sv for how this slots
// into the boundary from stage-control.md section 3, and everything
// ctrl_stage.sv still stubs out.

module ctrl_lr_stage
  import ctrl_pkg::*;
(
    input  logic                    clk,
    input  logic                    rst,

    // 3 raw LR sub-instructions for the current VLIW word (order matches
    // SLOT_COUNT["lr"] = 3 in instruction_spec.py: lr_inst_0, lr_inst_1,
    // lr_inst_2).
    input  logic [LR_INST_W-1:0]    lr_inst_i [LR_LANES],

    // CR read port -- external dependency on issue #4's ctrl_cr_regfile.
    input  logic [REG_WIDTH-1:0]    cr_rdata_i [CR_COUNT],

    // Post-this-cycle-write LR values, i.e. what stage-control.md section 5
    // calls the values CTRL forwards on the dispatch buses (issue #5) --
    // NOT the same as lr_rdata_snapshot_o below. Exposed combinationally so
    // issue #5's bus-packing logic can read "this cycle's post-LR-write LR
    // state" without waiting a cycle.
    output logic [REG_WIDTH-1:0]    lr_rdata_postwrite_o [LR_COUNT],

    // Pre-this-cycle-write LR values -- what CTRL's own reads (cond-slot
    // operands, per stage-control.md section 5) must use instead of the
    // post-write bus above. This is the regfile's registered state, sampled
    // before this cycle's writes land.
    output logic [REG_WIDTH-1:0]    lr_rdata_snapshot_o [LR_COUNT],

    output logic                    conflict_o
);

  logic                 lane_we    [LR_LANES];
  logic [LR_IDX_W-1:0]  lane_waddr [LR_LANES];
  logic [REG_WIDTH-1:0] lane_wdata [LR_LANES];

  // The register file's current (pre-this-cycle-write) output. Driven
  // straight through to lr_rdata_snapshot_o below: reads here are
  // combinational off flops that haven't yet latched this cycle's write,
  // which is exactly the read-before-write semantics
  // Ipu.execute_vliw_cycle fakes in software via RegFile.snapshot(). See
  // ctrl_lr_lane.sv's header comment.
  logic [REG_WIDTH-1:0] lr_rdata_snapshot [LR_COUNT];
  assign lr_rdata_snapshot_o = lr_rdata_snapshot;

  genvar g;
  generate
    for (g = 0; g < LR_LANES; g++) begin : gen_lane
      ctrl_lr_lane u_lane (
          .lr_inst_i (lr_inst_i[g]),
          .lr_rdata_i(lr_rdata_snapshot),
          .cr_rdata_i(cr_rdata_i),
          .lr_we_o   (lane_we[g]),
          .lr_waddr_o(lane_waddr[g]),
          .lr_wdata_o(lane_wdata[g])
      );
    end
  endgenerate

  ctrl_lr_regfile u_lr_regfile (
      .clk       (clk),
      .rst       (rst),
      .we_i      (lane_we),
      .waddr_i   (lane_waddr),
      .wdata_i   (lane_wdata),
      .lr_rdata_o(lr_rdata_snapshot),
      .conflict_o(conflict_o)
  );

  // Post-write forwarding value for LR register `idx`: if some active lane
  // wrote it this cycle, the post-write value is that lane's wdata (lowest
  // lane wins on a same-cycle conflict, matching ctrl_lr_regfile.sv's
  // policy); otherwise it's just the unwritten snapshot value. This
  // mirrors stage-control.md section 5's rule that downstream stages see
  // "this cycle's post-LR-write values," computed here without waiting for
  // the register file's own clocked write to complete.
  always_comb begin
    for (int unsigned idx = 0; idx < LR_COUNT; idx++) begin
      lr_rdata_postwrite_o[idx] = lr_rdata_snapshot[idx];
      for (int unsigned lane = LR_LANES; lane > 0; lane--) begin
        if (lane_we[lane-1] && (lane_waddr[lane-1] == LR_IDX_W'(idx))) begin
          lr_rdata_postwrite_o[idx] = lane_wdata[lane-1];
        end
      end
    end
  end

endmodule : ctrl_lr_stage
