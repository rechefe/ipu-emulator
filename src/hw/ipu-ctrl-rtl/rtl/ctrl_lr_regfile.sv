// ctrl_lr_regfile.sv — the 16x20-bit LR register file, with 3 write ports
// (one per ctrl_lr_lane instance) and full combinational read ports.
//
// Ground truth: RegFile.get_lr/set_lr (regfile.py), backed by set_scalar's
// LR_CR_SCALAR_VALUE_MASK (ipu_config.py: 20 bits) -- storage is exactly
// 20 bits wide, not the 32-bit width the `& 0xFFFFFFFF` masks visible in
// ipu.py's execute_lr_* bodies might suggest (those masks are vestigial:
// they're applied before set_lr(), which re-masks to 20 bits on the way
// into storage regardless).
//
// Conflict handling: Ipu._dispatch_lr_slots (ipu.py:557) raises a Python
// exception if two of the three lanes target the same destination LR in
// one VLIW word -- there is no defined "what wins" behavior in the
// emulator because it refuses to execute the cycle at all. Hardware has no
// equivalent of raising an exception mid-cycle, so this module picks an
// explicit, deterministic policy (lowest-numbered active lane wins) and
// exposes `conflict_o` so a conflict is at least observable, but this is a
// placeholder pending planning/ctrl-stage-rtl/03-lr-regfile-alu.md's open
// question: is this purely an assembler-time invariant (well-formed
// programs never encode it, so hardware doesn't need to arbitrate at all),
// or does CTRL need a real runtime guard? Do not treat the priority-write
// behavior below as a settled design decision.
//
// See planning/ctrl-stage-rtl/03-lr-regfile-alu.md.

module ctrl_lr_regfile
  import ctrl_pkg::*;
(
    input  logic                 clk,
    input  logic                 rst,        // synchronous, active-high -- stage-control.md section 3.1

    // One write request per lane, from ctrl_lr_lane.
    input  logic                 we_i    [LR_LANES],
    input  logic [LR_IDX_W-1:0]  waddr_i [LR_LANES],
    input  logic [REG_WIDTH-1:0] wdata_i [LR_LANES],

    // Combinational read port, one entry per LR register -- these are the
    // pre-this-cycle-write values every ctrl_lr_lane instance (and,
    // eventually, the dispatch-bus packing logic's *own* reads per
    // stage-control.md section 5) consumes as "the snapshot."
    output logic [REG_WIDTH-1:0] lr_rdata_o [LR_COUNT],

    // Diagnostic only -- see conflict-handling note above. Not meant to
    // gate anything downstream yet.
    output logic                 conflict_o
);

  logic [REG_WIDTH-1:0] regs [LR_COUNT];

  assign lr_rdata_o = regs;

  // Same-destination conflict across the 3 lanes (diagnostic only -- see
  // module header).
  always_comb begin
    conflict_o = 1'b0;
    for (int unsigned i = 0; i < LR_LANES; i++) begin
      for (int unsigned j = i + 1; j < LR_LANES; j++) begin
        if (we_i[i] && we_i[j] && (waddr_i[i] == waddr_i[j])) begin
          conflict_o = 1'b1;
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      for (int unsigned i = 0; i < LR_COUNT; i++) begin
        regs[i] <= '0;
      end
    end else begin
      // Lowest-numbered active lane wins a same-destination conflict --
      // see the "placeholder" note above; reverse iteration so lane 0's
      // assignment is the last one applied (i.e. wins).
      for (int unsigned lane = LR_LANES; lane > 0; lane--) begin
        if (we_i[lane-1]) begin
          regs[waddr_i[lane-1]] <= wdata_i[lane-1];
        end
      end
    end
  end

endmodule : ctrl_lr_regfile
