// A stand-in IPU with the ports of the real one (interfaces.md), for bringing
// up the platform before the IPU RTL exists. Build with IPU_RTL=stub.
//
// What it does:
// - The host port is the generated register block (ipu_host_regs), so the
//   register map, the AXI path and the driver are tested on the card.
// - IMEM windows, CR banks, PROG_LEN and FORMAT store what the host writes.
//   Beats 5 to 7 of an IMEM word read 0 and ignore writes.
// - ARM in IDLE, with the inactive context's tables valid, runs an empty
//   kernel: active_ctx flips, kernel_start pulses, and kernel_done pulses the
//   next cycle. The cache unit then sees a whole commit, done and drain cycle.
//
// What it does not do: execute words, touch XMEM, check access rules (every
// access answers OKAY), or accept any other command.
module ipu_top
  import ipu_socket_pkg::*;
#(
  parameter int unsigned HOST_ADDR_W = ipu_host_addr_pkg::ADDR_WIDTH
) (
  input  logic                   clk,
  input  logic                   rst_n,

  // Host AXI4 slave. The stub takes single beats with ID 0 (the platform's
  // AXI4-Lite bridge sends only those).
  input  logic                   s_axi_awvalid,
  output logic                   s_axi_awready,
  input  logic [HOST_ID_W-1:0]   s_axi_awid,
  input  logic [HOST_ADDR_W-1:0] s_axi_awaddr,
  input  logic [7:0]             s_axi_awlen,
  input  logic [2:0]             s_axi_awsize,
  input  logic [1:0]             s_axi_awburst,
  input  logic                   s_axi_wvalid,
  output logic                   s_axi_wready,
  input  logic [31:0]            s_axi_wdata,
  input  logic [3:0]             s_axi_wstrb,
  input  logic                   s_axi_wlast,
  output logic                   s_axi_bvalid,
  input  logic                   s_axi_bready,
  output logic [HOST_ID_W-1:0]   s_axi_bid,
  output logic [1:0]             s_axi_bresp,
  input  logic                   s_axi_arvalid,
  output logic                   s_axi_arready,
  input  logic [HOST_ID_W-1:0]   s_axi_arid,
  input  logic [HOST_ADDR_W-1:0] s_axi_araddr,
  input  logic [7:0]             s_axi_arlen,
  input  logic [2:0]             s_axi_arsize,
  input  logic [1:0]             s_axi_arburst,
  output logic                   s_axi_rvalid,
  input  logic                   s_axi_rready,
  output logic [HOST_ID_W-1:0]   s_axi_rid,
  output logic [31:0]            s_axi_rdata,
  output logic [1:0]             s_axi_rresp,
  output logic                   s_axi_rlast,

  output logic                   irq,

  // Cache unit
  input  tbl_view_t              tbl_view,
  input  logic [1:0]             tables_valid,
  output logic                   active_ctx,
  output logic [1:0]             ctx_locked,
  output logic                   kernel_start,
  output logic                   kernel_done,
  output logic                   kernel_abort,
  input  logic                   bank_pub_valid,
  input  logic [BANK_W-1:0]      bank_pub_id,
  input  logic [TAG_W-1:0]       bank_pub_tag,
  input  logic                   bank_free_valid,
  input  logic [BANK_W-1:0]      bank_free_id,
  output logic                   bank_req_valid,
  output logic [TAG_W-1:0]       bank_req_tag,
  output logic                   bank_req_write,
  output logic                   bank_release_valid,
  output logic [BANK_W-1:0]      bank_release_id,
  output logic                   bank_flush_valid,
  output logic [BANK_W-1:0]      bank_flush_id,

  // XMEM
  output logic                   ld_valid,
  output logic [BANK_W-1:0]      ld_bank,
  output logic [ROW_W-1:0]       ld_row,
  input  logic [ROW_BITS-1:0]    ld_data,
  input  logic [META_BITS-1:0]   ld_meta,
  output logic                   st_valid,
  output logic [BANK_W-1:0]      st_bank,
  output logic [ROW_W-1:0]       st_row,
  output logic [ROW_BITS-1:0]    st_data,
  output logic [META_BITS-1:0]   st_meta
);

  ipu_host_regs_pkg::ipu_host__in_t  hwif_in;
  ipu_host_regs_pkg::ipu_host__out_t hwif_out;

  ipu_host_regs u_regs (
    .clk            (clk),
    .arst_n         (rst_n),
    .s_axil_awready (s_axi_awready),
    .s_axil_awvalid (s_axi_awvalid),
    .s_axil_awaddr  (s_axi_awaddr),
    .s_axil_awprot  (3'b000),
    .s_axil_wready  (s_axi_wready),
    .s_axil_wvalid  (s_axi_wvalid),
    .s_axil_wdata   (s_axi_wdata),
    .s_axil_wstrb   (s_axi_wstrb),
    .s_axil_bready  (s_axi_bready),
    .s_axil_bvalid  (s_axi_bvalid),
    .s_axil_bresp   (s_axi_bresp),
    .s_axil_arready (s_axi_arready),
    .s_axil_arvalid (s_axi_arvalid),
    .s_axil_araddr  (s_axi_araddr),
    .s_axil_arprot  (3'b000),
    .s_axil_rready  (s_axi_rready),
    .s_axil_rvalid  (s_axi_rvalid),
    .s_axil_rdata   (s_axi_rdata),
    .s_axil_rresp   (s_axi_rresp),
    .hwif_in        (hwif_in),
    .hwif_out       (hwif_out)
  );

  assign s_axi_bid = '0;
  assign s_axi_rid = '0;
  assign s_axi_rlast = 1'b1;

  // ---- Storage behind the external registers --------------------------------

  logic [31:0] imem [2][8192];
  logic [31:0] cr [2][16];
  logic [10:0] prog_len [2];
  logic [3:0]  fmt [2];

  logic        imem_rd_ack [2];
  logic        imem_wr_ack [2];
  logic [31:0] imem_rd_data [2];

  // An IMEM window access is answered the cycle after its request.
  for (genvar c = 0; c < 2; c++) begin : g_imem
    logic        req, is_wr;
    logic [14:0] addr;
    logic [31:0] wr_data;
    assign req     = (c == 0) ? hwif_out.IMEM0.req       : hwif_out.IMEM1.req;
    assign is_wr   = (c == 0) ? hwif_out.IMEM0.req_is_wr : hwif_out.IMEM1.req_is_wr;
    assign addr    = (c == 0) ? hwif_out.IMEM0.addr      : hwif_out.IMEM1.addr;
    assign wr_data = (c == 0) ? hwif_out.IMEM0.wr_data   : hwif_out.IMEM1.wr_data;

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        imem_rd_ack[c] <= 1'b0;
        imem_wr_ack[c] <= 1'b0;
        imem_rd_data[c] <= '0;
      end else begin
        imem_rd_ack[c] <= req && !is_wr;
        imem_wr_ack[c] <= req && is_wr;
        // Beats 5 to 7 of a 32-byte word hold nothing (ADR 0007).
        imem_rd_data[c] <= (addr[4:2] < 3'd5) ? imem[c][addr[14:2]] : '0;
        if (req && is_wr && addr[4:2] < 3'd5) imem[c][addr[14:2]] <= wr_data;
      end
    end
  end

  logic ext_ack;          // any external register request, answered next cycle
  logic ext_is_wr;
  logic ext_req;

  always_comb begin
    ext_req = hwif_out.REGS.COMMAND.req || hwif_out.REGS.DEBUG.PC.req;
    for (int c = 0; c < 2; c++) begin
      ext_req |= hwif_out.REGS.PROG_LEN[c].req || hwif_out.REGS.FORMAT[c].req;
      for (int i = 0; i < 16; i++) ext_req |= hwif_out.REGS.CR_BANK[c].CR[i].req;
    end
  end

  // ---- The empty kernel ------------------------------------------------------

  logic arm, done_next;
  assign arm = hwif_out.REGS.COMMAND.req && hwif_out.REGS.COMMAND.req_is_wr &&
               hwif_out.REGS.COMMAND.wr_data.ARM && tables_valid[!active_ctx];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ext_ack      <= 1'b0;
      ext_is_wr    <= 1'b0;
      active_ctx   <= 1'b0;       // IPU-IF-3
      kernel_start <= 1'b0;
      kernel_done  <= 1'b0;
      done_next    <= 1'b0;
      for (int c = 0; c < 2; c++) begin
        prog_len[c] <= '0;
        fmt[c] <= '0;
        for (int i = 0; i < 16; i++) cr[c][i] <= (i == 1) ? 32'd1 : 32'd0;
      end
    end else begin
      ext_ack      <= ext_req;
      ext_is_wr    <= ext_req && (
          hwif_out.REGS.COMMAND.req_is_wr || hwif_out.REGS.DEBUG.PC.req_is_wr ||
          hwif_out.REGS.PROG_LEN[0].req_is_wr || hwif_out.REGS.PROG_LEN[1].req_is_wr ||
          hwif_out.REGS.FORMAT[0].req_is_wr || hwif_out.REGS.FORMAT[1].req_is_wr ||
          cr_req_is_wr());
      kernel_start <= arm;
      done_next    <= arm;
      kernel_done  <= done_next;
      if (arm) active_ctx <= !active_ctx;
      for (int c = 0; c < 2; c++) begin
        if (hwif_out.REGS.PROG_LEN[c].req && hwif_out.REGS.PROG_LEN[c].req_is_wr)
          prog_len[c] <= hwif_out.REGS.PROG_LEN[c].wr_data.WORDS;
        if (hwif_out.REGS.FORMAT[c].req && hwif_out.REGS.FORMAT[c].req_is_wr)
          fmt[c] <= hwif_out.REGS.FORMAT[c].wr_data.FMT;
        for (int i = 2; i < 16; i++)
          if (hwif_out.REGS.CR_BANK[c].CR[i].req && hwif_out.REGS.CR_BANK[c].CR[i].req_is_wr)
            cr[c][i] <= hwif_out.REGS.CR_BANK[c].CR[i].wr_data.VALUE;
      end
    end
  end

  function automatic logic cr_req_is_wr();
    logic any = 1'b0;
    for (int c = 0; c < 2; c++)
      for (int i = 0; i < 16; i++)
        any |= hwif_out.REGS.CR_BANK[c].CR[i].req && hwif_out.REGS.CR_BANK[c].CR[i].req_is_wr;
    return any;
  endfunction

  // ---- Register block inputs ---------------------------------------------------

  always_comb begin
    hwif_in = '{default: '0};
    hwif_in.IMEM0.rd_ack  = imem_rd_ack[0];
    hwif_in.IMEM0.wr_ack  = imem_wr_ack[0];
    hwif_in.IMEM0.rd_data = imem_rd_data[0];
    hwif_in.IMEM1.rd_ack  = imem_rd_ack[1];
    hwif_in.IMEM1.wr_ack  = imem_wr_ack[1];
    hwif_in.IMEM1.rd_data = imem_rd_data[1];

    hwif_in.REGS.PARAMS.IMEM_DEPTH_LOG2.next = 5'd10;
    hwif_in.REGS.PARAMS.BREAKPOINTS.next     = 4'd4;
    hwif_in.REGS.PARAMS.LANES.next           = 8'd128;
    hwif_in.REGS.DEBUG.DBG_CAP.BREAKPOINTS.next = 4'd4;

    hwif_in.REGS.STATUS.ACTIVE_CTX.next   = active_ctx;
    hwif_in.REGS.STATUS.TABLES_VALID.next = tables_valid;
    hwif_in.REGS.INT_STATUS.KERNEL_START.hwset = kernel_start;
    hwif_in.REGS.INT_STATUS.KERNEL_DONE.hwset  = kernel_done;

    // External registers: every request is answered the next cycle. Reads
    // return the stored value of whichever register was addressed.
    hwif_in.REGS.COMMAND.wr_ack = ext_ack && ext_is_wr;
    hwif_in.REGS.DEBUG.PC.rd_ack = ext_ack && !ext_is_wr;
    hwif_in.REGS.DEBUG.PC.wr_ack = ext_ack && ext_is_wr;
    for (int c = 0; c < 2; c++) begin
      hwif_in.REGS.PROG_LEN[c].rd_ack = ext_ack && !ext_is_wr;
      hwif_in.REGS.PROG_LEN[c].wr_ack = ext_ack && ext_is_wr;
      hwif_in.REGS.PROG_LEN[c].rd_data.WORDS = prog_len[c];
      hwif_in.REGS.FORMAT[c].rd_ack = ext_ack && !ext_is_wr;
      hwif_in.REGS.FORMAT[c].wr_ack = ext_ack && ext_is_wr;
      hwif_in.REGS.FORMAT[c].rd_data.FMT = fmt[c];
      for (int i = 0; i < 16; i++) begin
        hwif_in.REGS.CR_BANK[c].CR[i].rd_ack = ext_ack && !ext_is_wr;
        hwif_in.REGS.CR_BANK[c].CR[i].wr_ack = ext_ack && ext_is_wr;
        hwif_in.REGS.CR_BANK[c].CR[i].rd_data.VALUE = cr[c][i];
      end
    end
  end

  assign irq = |({hwif_out.REGS.INT_STATUS.HOST_ERR.value, hwif_out.REGS.INT_STATUS.HALTED.value,
                  hwif_out.REGS.INT_STATUS.KERNEL_DONE.value, hwif_out.REGS.INT_STATUS.KERNEL_START.value} &
                 {hwif_out.REGS.INT_ENABLE.HOST_ERR.value, hwif_out.REGS.INT_ENABLE.HALTED.value,
                  hwif_out.REGS.INT_ENABLE.KERNEL_DONE.value, hwif_out.REGS.INT_ENABLE.KERNEL_START.value});

  // ---- Ports the stub does not drive -------------------------------------------

  assign ctx_locked         = 2'b00;
  assign kernel_abort       = 1'b0;
  assign bank_req_valid     = 1'b0;
  assign bank_req_tag       = '0;
  assign bank_req_write     = 1'b0;
  assign bank_release_valid = 1'b0;
  assign bank_release_id    = '0;
  assign bank_flush_valid   = 1'b0;
  assign bank_flush_id      = '0;
  assign ld_valid           = 1'b0;
  assign ld_bank            = '0;
  assign ld_row             = '0;
  assign st_valid           = 1'b0;
  assign st_bank            = '0;
  assign st_row             = '0;
  assign st_data            = '0;
  assign st_meta            = '0;

  // Unused inputs: tbl_view, bank events, ld_data, ld_meta, AXI4 burst fields.
  logic unused;
  assign unused = ^{tbl_view, bank_pub_valid, bank_pub_id, bank_pub_tag, bank_free_valid,
                    bank_free_id, ld_data, ld_meta, s_axi_awid, s_axi_awlen, s_axi_awsize,
                    s_axi_awburst, s_axi_wlast, s_axi_arid, s_axi_arlen, s_axi_arsize,
                    s_axi_arburst, cr[0][0], cr[1][0]};

endmodule
