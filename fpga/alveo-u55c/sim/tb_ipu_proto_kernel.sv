// Simulation of the prototype kernel with the stub IPU: the host path, the
// cache unit's commit, fill, load, store, done and drain, and the stub's
// empty kernel. Run with sim/run.sh (Verilator 5).
`timescale 1ns / 1ps

module tb_ipu_proto_kernel;
  import ipu_socket_pkg::*;

  localparam int unsigned AW = 18;
  localparam logic [AW-1:0] PLAT = 18'h20000;
  localparam logic [63:0] DRAM = 64'h0010_0000;
  localparam int unsigned BANK_BYTES = 128 * 1024;

  logic clk = 1'b0;
  logic rst_n = 1'b0;
  always #5 clk = !clk;

  // AXI4-Lite control
  logic          awvalid = 0, awready, wvalid = 0, wready, bvalid, bready = 0;
  logic [AW-1:0] awaddr = '0, araddr = '0;
  logic [31:0]   wdata = '0, rdata;
  logic [1:0]    bresp, rresp;
  logic          arvalid = 0, arready, rvalid, rready = 0;

  // AXI4 HBM
  logic         hbm_awvalid, hbm_awready, hbm_wvalid, hbm_wready, hbm_wlast;
  logic         hbm_bvalid, hbm_bready, hbm_arvalid, hbm_arready, hbm_rvalid, hbm_rready, hbm_rlast;
  logic [63:0]  hbm_awaddr, hbm_araddr;
  logic [7:0]   hbm_awlen, hbm_arlen;
  logic [511:0] hbm_wdata, hbm_rdata;

  ipu_proto_kernel dut (
    .ap_clk(clk), .ap_rst_n(rst_n),
    .s_axi_control_awvalid(awvalid), .s_axi_control_awready(awready), .s_axi_control_awaddr(awaddr),
    .s_axi_control_wvalid(wvalid), .s_axi_control_wready(wready), .s_axi_control_wdata(wdata),
    .s_axi_control_wstrb(4'hF), .s_axi_control_bvalid(bvalid), .s_axi_control_bready(bready),
    .s_axi_control_bresp(bresp), .s_axi_control_arvalid(arvalid), .s_axi_control_arready(arready),
    .s_axi_control_araddr(araddr), .s_axi_control_rvalid(rvalid), .s_axi_control_rready(rready),
    .s_axi_control_rdata(rdata), .s_axi_control_rresp(rresp),
    .m_axi_hbm_awvalid(hbm_awvalid), .m_axi_hbm_awready(hbm_awready), .m_axi_hbm_awaddr(hbm_awaddr),
    .m_axi_hbm_awid(), .m_axi_hbm_awlen(hbm_awlen), .m_axi_hbm_awsize(), .m_axi_hbm_awburst(),
    .m_axi_hbm_wvalid(hbm_wvalid), .m_axi_hbm_wready(hbm_wready), .m_axi_hbm_wdata(hbm_wdata),
    .m_axi_hbm_wstrb(), .m_axi_hbm_wlast(hbm_wlast), .m_axi_hbm_bvalid(hbm_bvalid),
    .m_axi_hbm_bready(hbm_bready), .m_axi_hbm_bresp(2'b00), .m_axi_hbm_bid(1'b0),
    .m_axi_hbm_arvalid(hbm_arvalid), .m_axi_hbm_arready(hbm_arready), .m_axi_hbm_araddr(hbm_araddr),
    .m_axi_hbm_arid(), .m_axi_hbm_arlen(hbm_arlen), .m_axi_hbm_arsize(), .m_axi_hbm_arburst(),
    .m_axi_hbm_rvalid(hbm_rvalid), .m_axi_hbm_rready(hbm_rready), .m_axi_hbm_rdata(hbm_rdata),
    .m_axi_hbm_rresp(2'b00), .m_axi_hbm_rlast(hbm_rlast), .m_axi_hbm_rid(1'b0)
  );

  // ---- HBM: 64-byte words, one burst at a time, a few cycles of latency ----

  logic [511:0] hbm [logic [63:0]];
  logic [63:0]  rd_addr, wr_addr;
  int           rd_left, wr_wait;
  logic         rd_busy = 0, wr_busy = 0, b_pending = 0;

  function automatic logic [511:0] hbm_word(logic [63:0] addr);
    return hbm.exists(addr >> 6) ? hbm[addr >> 6] : '0;
  endfunction

  assign hbm_arready = !rd_busy;
  assign hbm_rvalid  = rd_busy && wr_wait == 0;
  assign hbm_rdata   = hbm_word(rd_addr);
  assign hbm_rlast   = rd_left == 1;
  assign hbm_awready = !wr_busy && !b_pending;
  assign hbm_wready  = wr_busy;
  assign hbm_bvalid  = b_pending;

  always_ff @(posedge clk) begin
    if (hbm_arvalid && hbm_arready) begin
      rd_busy <= 1;
      rd_addr <= hbm_araddr;
      rd_left <= int'(hbm_arlen) + 1;
      wr_wait <= 3;
    end else if (rd_busy && wr_wait != 0) begin
      wr_wait <= wr_wait - 1;
    end else if (hbm_rvalid && hbm_rready) begin
      rd_addr <= rd_addr + 64;
      rd_left <= rd_left - 1;
      if (rd_left == 1) rd_busy <= 0;
    end
    if (hbm_awvalid && hbm_awready) begin
      wr_busy <= 1;
      wr_addr <= hbm_awaddr;
    end else if (hbm_wvalid && hbm_wready) begin
      hbm[wr_addr >> 6] <= hbm_wdata;
      wr_addr <= wr_addr + 64;
      if (hbm_wlast) begin
        wr_busy <= 0;
        b_pending <= 1;
      end
    end
    if (hbm_bvalid && hbm_bready) b_pending <= 0;
  end

  // ---- Host accesses ----

  task automatic write32(input logic [AW-1:0] addr, input logic [31:0] data);
    @(negedge clk);
    awaddr = addr; awvalid = 1; wdata = data; wvalid = 1; bready = 1;
    while (awvalid || wvalid) begin
      @(posedge clk);
      if (awready) awvalid = 0;
      if (wready) wvalid = 0;
      @(negedge clk);
    end
    while (!bvalid) @(negedge clk);
    @(posedge clk);
    @(negedge clk);
    bready = 0;
  endtask

  task automatic read32(input logic [AW-1:0] addr, output logic [31:0] data);
    @(negedge clk);
    araddr = addr; arvalid = 1; rready = 1;
    while (arvalid) begin
      @(posedge clk);
      if (arready) arvalid = 0;
      @(negedge clk);
    end
    while (!rvalid) @(negedge clk);
    data = rdata;
    @(posedge clk);
    @(negedge clk);
    rready = 0;
  endtask

  int errors = 0;
  task automatic expect32(input string what, input logic [AW-1:0] addr, input logic [31:0] want);
    logic [31:0] got;
    read32(addr, got);
    if (got !== want) begin
      $display("FAIL %s: 0x%08h, want 0x%08h", what, got, want);
      errors++;
    end else begin
      $display("ok   %s", what);
    end
  endtask

  task automatic wait_field(input string what, input logic [AW-1:0] addr, input logic [31:0] mask,
                            input logic [31:0] want);
    logic [31:0] got;
    for (int i = 0; i < 2000; i++) begin
      read32(addr, got);
      if ((got & mask) == want) begin
        $display("ok   %s", what);
        return;
      end
      repeat (50) @(posedge clk);
    end
    $display("FAIL %s: timed out, last 0x%08h", what, got);
    errors++;
  endtask

  // Platform offsets (proto_platform.rdl).
  localparam logic [AW-1:0] CTRL = PLAT + 'h010, COMMIT = PLAT + 'h020, CU_STATUS = PLAT + 'h024;
  localparam logic [AW-1:0] BANK0 = PLAT + 'h040, EX_CMD = PLAT + 'h100, EX_ADDR = PLAT + 'h104;
  localparam logic [AW-1:0] EX_CTX = PLAT + 'h108, EX_DATA = PLAT + 'h180;
  function automatic logic [AW-1:0] table_reg(int set, int t, int off);
    return PLAT + AW'('h1000 + set * 'h200 + t * 'h20 + off);
  endfunction

  // Row r of page p in DRAM: byte i is (p * 7 + r + i) mod 256.
  function automatic logic [1023:0] image_row(int p, int r);
    logic [1023:0] row;
    for (int i = 0; i < 128; i++) row[8*i +: 8] = 8'(p * 7 + r + i);
    return row;
  endfunction

  initial begin
    logic [31:0] got;
    logic [1023:0] row;
    for (int p = 0; p < 2; p++)
      for (int r = 0; r < 1024; r++) begin
        row = image_row(p, r);
        hbm[(DRAM + 64'(p * BANK_BYTES + r * 128)) >> 6] = row[511:0];
        hbm[(DRAM + 64'(p * BANK_BYTES + r * 128 + 64)) >> 6] = row[1023:512];
      end

    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (5) @(posedge clk);

    // Identification and the IMEM window (32-byte stride, beats 5-7 read 0).
    expect32("IPU ID", 18'h10000, 32'h49505548);
    expect32("platform ID", PLAT, 32'h49505550);
    write32(18'h08000 + 32 * 3 + 4 * 4, 32'h0000_0123);
    write32(18'h08000 + 32 * 3 + 4 * 5, 32'hDEAD_BEEF);
    expect32("IMEM1 word 3 beat 4", 18'h08000 + 32 * 3 + 4 * 4, 32'h0000_0123);
    expect32("IMEM1 word 3 beat 5 reads 0", 18'h08000 + 32 * 3 + 4 * 5, 32'h0);
    write32(18'h10148, 32'd77);
    expect32("CR_BANK[1].CR[2]", 18'h10148, 32'd77);
    expect32("CR_BANK[0].CR[1]", 18'h10104, 32'd1);

    // Table set 1, table 0: scratch pad over banks 0 and 1.
    write32(table_reg(1, 0, 'h00), 32'h7);            // ENABLE, TYPE = SCRATCH_PAD
    write32(table_reg(1, 0, 'h04), 32'h0);            // FIRST_BANK 0
    write32(table_reg(1, 0, 'h08), DRAM[31:0]);
    write32(table_reg(1, 0, 'h0C), DRAM[63:32]);
    write32(table_reg(1, 0, 'h10), 32'd2);            // SIZE 2 pages
    write32(COMMIT, 32'h2);
    expect32("commit accepted, tables_valid[1]", CU_STATUS, 32'h0000_0006);  // DRAINED[0] stays 1
    wait_field("bank 1 VALID, tag {1, 0, 1}", BANK0 + 4, 32'h7FFF_0003, {1'b0, 15'h4001, 14'h0, 2'd2});
    expect32("bank 0 VALID, tag {1, 0, 0}", BANK0, {1'b0, 15'h4000, 14'h0, 2'd2});

    // The exerciser loads row 5 of bank 0 and stores a pattern to row 7 of bank 1.
    write32(CTRL, 32'h4);
    write32(EX_CTX, 32'h1);                           // active_ctx 1
    write32(EX_ADDR, {18'h0, 10'd5, 4'd0});
    write32(EX_CMD, 32'h1);                           // LOAD
    row = image_row(0, 5);
    expect32("load: row 5 word 0", EX_DATA, row[31:0]);
    expect32("load: row 5 word 31", EX_DATA + 31 * 4, row[1023:992]);
    for (int i = 0; i < 32; i++) write32(EX_DATA + AW'(4 * i), 32'hA5A5_0000 + i);
    write32(EX_ADDR, {18'h0, 10'd7, 4'd1});
    write32(EX_CMD, 32'h2);                           // STORE
    expect32("bank 1 dirty", BANK0 + 4, {1'b0, 15'h4001, 13'h0, 1'b1, 2'd2});
    write32(EX_CMD, 32'h20);                          // KERNEL_DONE
    wait_field("drained[1]", CU_STATUS, 32'hF, 32'h0000_000C);
    expect32("bank 0 free after done", BANK0, {1'b0, 15'h4000, 14'h0, 2'd0});
    begin
      logic [63:0] at;
      at = DRAM + 64'(BANK_BYTES + 7 * 128);
      if (hbm[at >> 6][31:0] !== 32'hA5A5_0000 || hbm[(at + 64) >> 6][511:480] !== 32'hA5A5_001F) begin
        $display("FAIL drain: page 1 row 7 not written back");
        errors++;
      end else $display("ok   drain wrote page 1 row 7");
      at = DRAM + 64'(BANK_BYTES + 8 * 128);
      row = image_row(1, 8);
      if (hbm[at >> 6] !== row[511:0]) begin
        $display("FAIL drain: page 1 row 8 changed");
        errors++;
      end else $display("ok   drain kept page 1 row 8");
    end

    // The stub IPU: commit set 1 again, ARM, and the empty kernel runs.
    write32(CTRL, 32'h0);
    write32(COMMIT, 32'h2);
    write32(18'h10010, 32'h1);                        // COMMAND.ARM
    wait_field("ARM: active_ctx 1", 18'h10014, 32'h4, 32'h4);
    expect32("interrupt status: start and done", 18'h10020, 32'h3);
    wait_field("drained[1] after the empty kernel", CU_STATUS, 32'hF, 32'h0000_000C);

    if (errors == 0) $display("PASS");
    else $display("FAILED: %0d", errors);
    $finish;
  end

  initial begin
    #50ms;
    $display("FAIL: timeout");
    $finish;
  end
endmodule
