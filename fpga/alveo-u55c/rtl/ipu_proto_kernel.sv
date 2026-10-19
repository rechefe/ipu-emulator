// The FPGA prototype: a Vitis user-managed RTL kernel for the Alveo U55C
// (fpga-prototype.md). It holds the IPU (`ipu_top`: the RTL repository's, or
// ipu_top_stub.sv), the prototype cache unit and the platform registers.
//
// Control address map (s_axi_control, 256 KiB):
//   0x00000-0x1FFFF  the IPU host port (ipu_host.rdl)
//   0x20000-0x3FFFF  the platform (proto_platform.rdl)
module ipu_proto_kernel
  import ipu_socket_pkg::*;
  import proto_pkg::*;
#(
  parameter int unsigned C_S_AXI_CONTROL_ADDR_WIDTH = 18,
  parameter int unsigned C_M_AXI_HBM_ADDR_WIDTH = 64,
  parameter int unsigned C_M_AXI_HBM_DATA_WIDTH = 512,
  parameter int unsigned C_M_AXI_HBM_ID_WIDTH = 1,
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int unsigned CLK_KHZ = 100_000
) (
  input  logic                                  ap_clk,
  input  logic                                  ap_rst_n,

  // AXI4-Lite control (XRT: xrt::ip read_register / write_register)
  input  logic                                  s_axi_control_awvalid,
  output logic                                  s_axi_control_awready,
  input  logic [C_S_AXI_CONTROL_ADDR_WIDTH-1:0] s_axi_control_awaddr,
  input  logic                                  s_axi_control_wvalid,
  output logic                                  s_axi_control_wready,
  input  logic [31:0]                           s_axi_control_wdata,
  input  logic [3:0]                            s_axi_control_wstrb,
  output logic                                  s_axi_control_bvalid,
  input  logic                                  s_axi_control_bready,
  output logic [1:0]                            s_axi_control_bresp,
  input  logic                                  s_axi_control_arvalid,
  output logic                                  s_axi_control_arready,
  input  logic [C_S_AXI_CONTROL_ADDR_WIDTH-1:0] s_axi_control_araddr,
  output logic                                  s_axi_control_rvalid,
  input  logic                                  s_axi_control_rready,
  output logic [31:0]                           s_axi_control_rdata,
  output logic [1:0]                            s_axi_control_rresp,

  // AXI4 master to HBM
  output logic                                  m_axi_hbm_awvalid,
  input  logic                                  m_axi_hbm_awready,
  output logic [C_M_AXI_HBM_ADDR_WIDTH-1:0]     m_axi_hbm_awaddr,
  output logic [C_M_AXI_HBM_ID_WIDTH-1:0]       m_axi_hbm_awid,
  output logic [7:0]                            m_axi_hbm_awlen,
  output logic [2:0]                            m_axi_hbm_awsize,
  output logic [1:0]                            m_axi_hbm_awburst,
  output logic                                  m_axi_hbm_wvalid,
  input  logic                                  m_axi_hbm_wready,
  output logic [C_M_AXI_HBM_DATA_WIDTH-1:0]     m_axi_hbm_wdata,
  output logic [C_M_AXI_HBM_DATA_WIDTH/8-1:0]   m_axi_hbm_wstrb,
  output logic                                  m_axi_hbm_wlast,
  input  logic                                  m_axi_hbm_bvalid,
  output logic                                  m_axi_hbm_bready,
  input  logic [1:0]                            m_axi_hbm_bresp,
  input  logic [C_M_AXI_HBM_ID_WIDTH-1:0]       m_axi_hbm_bid,
  output logic                                  m_axi_hbm_arvalid,
  input  logic                                  m_axi_hbm_arready,
  output logic [C_M_AXI_HBM_ADDR_WIDTH-1:0]     m_axi_hbm_araddr,
  output logic [C_M_AXI_HBM_ID_WIDTH-1:0]       m_axi_hbm_arid,
  output logic [7:0]                            m_axi_hbm_arlen,
  output logic [2:0]                            m_axi_hbm_arsize,
  output logic [1:0]                            m_axi_hbm_arburst,
  input  logic                                  m_axi_hbm_rvalid,
  output logic                                  m_axi_hbm_rready,
  input  logic [C_M_AXI_HBM_DATA_WIDTH-1:0]     m_axi_hbm_rdata,
  input  logic [1:0]                            m_axi_hbm_rresp,
  input  logic                                  m_axi_hbm_rlast,
  input  logic [C_M_AXI_HBM_ID_WIDTH-1:0]       m_axi_hbm_rid
);

  localparam int IPU_ADDR_W = ipu_host_addr_pkg::ADDR_WIDTH;
  localparam int PLAT_ADDR_W = proto_platform_addr_pkg::ADDR_WIDTH;

  // ---- Resets ------------------------------------------------------------------
  // ap_rst_n is synchronous to ap_clk. The platform registers' soft resets
  // assert asynchronously and release through two flops (IPU-IF-2).

  logic plat_rst_n, ipu_rst_n, cu_rst_n;
  logic ipu_reset_req, cu_reset_req;
  logic [1:0] ipu_rst_sync, cu_rst_sync;

  assign plat_rst_n = ap_rst_n;

  always_ff @(posedge ap_clk or negedge plat_rst_n) begin
    if (!plat_rst_n) begin
      ipu_rst_sync <= '0;
      cu_rst_sync  <= '0;
    end else begin
      ipu_rst_sync <= {ipu_rst_sync[0], !ipu_reset_req};
      cu_rst_sync  <= {cu_rst_sync[0], !cu_reset_req};
    end
  end
  assign ipu_rst_n = ipu_rst_sync[1];
  assign cu_rst_n  = cu_rst_sync[1];

  // ---- Control routing: bit 17 picks the IPU (0) or the platform (1) --------------

  logic                   ipu_awvalid, ipu_awready, ipu_wvalid, ipu_wready, ipu_bvalid, ipu_bready;
  logic                   ipu_arvalid, ipu_arready, ipu_rvalid, ipu_rready;
  logic [1:0]             ipu_bresp, ipu_rresp;
  logic [31:0]            ipu_rdata;
  logic                   plat_awvalid, plat_awready, plat_wvalid, plat_wready, plat_bvalid, plat_bready;
  logic                   plat_arvalid, plat_arready, plat_rvalid, plat_rready;
  logic [1:0]             plat_bresp, plat_rresp;
  logic [31:0]            plat_rdata;

  axil_split2 #(.ADDR_W(C_S_AXI_CONTROL_ADDR_WIDTH), .SEL_BIT(17)) u_split (
    .clk      (ap_clk),
    .rst_n    (plat_rst_n),
    .s_awvalid(s_axi_control_awvalid), .s_awready(s_axi_control_awready), .s_awaddr(s_axi_control_awaddr),
    .s_wvalid (s_axi_control_wvalid),  .s_wready (s_axi_control_wready),
    .s_bvalid (s_axi_control_bvalid),  .s_bready (s_axi_control_bready),  .s_bresp(s_axi_control_bresp),
    .s_arvalid(s_axi_control_arvalid), .s_arready(s_axi_control_arready), .s_araddr(s_axi_control_araddr),
    .s_rvalid (s_axi_control_rvalid),  .s_rready (s_axi_control_rready),
    .s_rdata  (s_axi_control_rdata),   .s_rresp  (s_axi_control_rresp),
    .m0_awvalid(ipu_awvalid), .m0_awready(ipu_awready), .m0_wvalid(ipu_wvalid), .m0_wready(ipu_wready),
    .m0_bvalid (ipu_bvalid),  .m0_bready (ipu_bready),  .m0_bresp (ipu_bresp),
    .m0_arvalid(ipu_arvalid), .m0_arready(ipu_arready), .m0_rvalid(ipu_rvalid), .m0_rready(ipu_rready),
    .m0_rdata  (ipu_rdata),   .m0_rresp  (ipu_rresp),
    .m1_awvalid(plat_awvalid), .m1_awready(plat_awready), .m1_wvalid(plat_wvalid), .m1_wready(plat_wready),
    .m1_bvalid (plat_bvalid),  .m1_bready (plat_bready),  .m1_bresp (plat_bresp),
    .m1_arvalid(plat_arvalid), .m1_arready(plat_arready), .m1_rvalid(plat_rvalid), .m1_rready(plat_rready),
    .m1_rdata  (plat_rdata),   .m1_rresp  (plat_rresp)
  );

  // ---- Platform registers --------------------------------------------------------

  proto_platform_regs_pkg::proto_platform__in_t  plat_in;
  proto_platform_regs_pkg::proto_platform__out_t plat_out;

  proto_platform_regs u_plat_regs (
    .clk            (ap_clk),
    .arst_n         (plat_rst_n),
    .s_axil_awready (plat_awready),
    .s_axil_awvalid (plat_awvalid),
    .s_axil_awaddr  (s_axi_control_awaddr[PLAT_ADDR_W-1:0]),
    .s_axil_awprot  (3'b000),
    .s_axil_wready  (plat_wready),
    .s_axil_wvalid  (plat_wvalid),
    .s_axil_wdata   (s_axi_control_wdata),
    .s_axil_wstrb   (s_axi_control_wstrb),
    .s_axil_bready  (plat_bready),
    .s_axil_bvalid  (plat_bvalid),
    .s_axil_bresp   (plat_bresp),
    .s_axil_arready (plat_arready),
    .s_axil_arvalid (plat_arvalid),
    .s_axil_araddr  (s_axi_control_araddr[PLAT_ADDR_W-1:0]),
    .s_axil_arprot  (3'b000),
    .s_axil_rready  (plat_rready),
    .s_axil_rvalid  (plat_rvalid),
    .s_axil_rdata   (plat_rdata),
    .s_axil_rresp   (plat_rresp),
    .hwif_in        (plat_in),
    .hwif_out       (plat_out)
  );

  assign ipu_reset_req = plat_out.CTRL.IPU_RESET.value;
  assign cu_reset_req  = plat_out.CTRL.CU_RESET.value;

  // ---- The IPU ---------------------------------------------------------------------

  tbl_view_t             tbl_view;
  logic [1:0]            tables_valid, ctx_locked, ipu_ctx_locked;
  logic                  active_ctx, ipu_active_ctx, irq;
  logic                  kernel_start, kernel_done, kernel_abort;
  logic                  ipu_kernel_start, ipu_kernel_done, ipu_kernel_abort;
  logic                  bank_pub_valid, bank_free_valid;
  logic [BANK_W-1:0]     bank_pub_id, bank_free_id;
  logic [TAG_W-1:0]      bank_pub_tag;
  logic                  bank_req_valid, bank_req_write, bank_release_valid, bank_flush_valid;
  logic [TAG_W-1:0]      bank_req_tag;
  logic [BANK_W-1:0]     bank_release_id, bank_flush_id;
  logic                  ipu_bank_req_valid, ipu_bank_req_write, ipu_bank_release_valid, ipu_bank_flush_valid;
  logic [TAG_W-1:0]      ipu_bank_req_tag;
  logic [BANK_W-1:0]     ipu_bank_release_id, ipu_bank_flush_id;
  logic                  ld_valid, st_valid, ipu_ld_valid, ipu_st_valid;
  logic [BANK_W-1:0]     ld_bank, st_bank, ipu_ld_bank, ipu_st_bank;
  logic [ROW_W-1:0]      ld_row, st_row, ipu_ld_row, ipu_st_row;
  logic [ROW_BITS-1:0]   ld_data, st_data, ipu_st_data;
  logic [META_BITS-1:0]  ld_meta, st_meta, ipu_st_meta;

  // The bridge sends single beats with ID 0, so the response IDs and RLAST carry nothing.
  logic [HOST_ID_W-1:0]  ipu_bid, ipu_rid;
  logic                  ipu_rlast;

  ipu_top u_ipu (
    .clk               (ap_clk),
    .rst_n             (ipu_rst_n),
    // AXI4-Lite into the IPU's AXI4 slave: single beats, ID 0.
    .s_axi_awvalid     (ipu_awvalid),
    .s_axi_awready     (ipu_awready),
    .s_axi_awid        ('0),
    .s_axi_awaddr      (s_axi_control_awaddr[IPU_ADDR_W-1:0]),
    .s_axi_awlen       (8'd0),
    .s_axi_awsize      (3'd2),
    .s_axi_awburst     (2'b01),
    .s_axi_wvalid      (ipu_wvalid),
    .s_axi_wready      (ipu_wready),
    .s_axi_wdata       (s_axi_control_wdata),
    .s_axi_wstrb       (s_axi_control_wstrb),
    .s_axi_wlast       (1'b1),
    .s_axi_bvalid      (ipu_bvalid),
    .s_axi_bready      (ipu_bready),
    .s_axi_bid         (ipu_bid),
    .s_axi_bresp       (ipu_bresp),
    .s_axi_arvalid     (ipu_arvalid),
    .s_axi_arready     (ipu_arready),
    .s_axi_arid        ('0),
    .s_axi_araddr      (s_axi_control_araddr[IPU_ADDR_W-1:0]),
    .s_axi_arlen       (8'd0),
    .s_axi_arsize      (3'd2),
    .s_axi_arburst     (2'b01),
    .s_axi_rvalid      (ipu_rvalid),
    .s_axi_rready      (ipu_rready),
    .s_axi_rdata       (ipu_rdata),
    .s_axi_rid         (ipu_rid),
    .s_axi_rresp       (ipu_rresp),
    .s_axi_rlast       (ipu_rlast),
    .irq               (irq),
    .tbl_view          (tbl_view),
    .tables_valid      (tables_valid),
    .active_ctx        (ipu_active_ctx),
    .ctx_locked        (ipu_ctx_locked),
    .kernel_start      (ipu_kernel_start),
    .kernel_done       (ipu_kernel_done),
    .kernel_abort      (ipu_kernel_abort),
    .bank_pub_valid    (bank_pub_valid),
    .bank_pub_id       (bank_pub_id),
    .bank_pub_tag      (bank_pub_tag),
    .bank_free_valid   (bank_free_valid),
    .bank_free_id      (bank_free_id),
    .bank_req_valid    (ipu_bank_req_valid),
    .bank_req_tag      (ipu_bank_req_tag),
    .bank_req_write    (ipu_bank_req_write),
    .bank_release_valid(ipu_bank_release_valid),
    .bank_release_id   (ipu_bank_release_id),
    .bank_flush_valid  (ipu_bank_flush_valid),
    .bank_flush_id     (ipu_bank_flush_id),
    .ld_valid          (ipu_ld_valid),
    .ld_bank           (ipu_ld_bank),
    .ld_row            (ipu_ld_row),
    .ld_data           (ld_data),
    .ld_meta           (ld_meta),
    .st_valid          (ipu_st_valid),
    .st_bank           (ipu_st_bank),
    .st_row            (ipu_st_row),
    .st_data           (ipu_st_data),
    .st_meta           (ipu_st_meta)
  );

  // ---- Socket exerciser: the host drives the cache unit's IPU-side ports -----------

  logic ex_on;
  logic ex_load_q;
  logic [ROW_BITS-1:0] ex_data;

  assign ex_on = plat_out.CTRL.EXERCISER.value;
  always_comb
    for (int i = 0; i < 32; i++) ex_data[32*i +: 32] = plat_out.EX_DATA[i].VALUE.value;

  always_ff @(posedge ap_clk or negedge plat_rst_n)
    if (!plat_rst_n) ex_load_q <= 1'b0;
    else ex_load_q <= ex_on && plat_out.EX_CMD.LOAD.value;

  always_comb begin
    if (ex_on) begin
      active_ctx         = plat_out.EX_CTX.ACTIVE_CTX.value;
      ctx_locked         = plat_out.EX_CTX.CTX_LOCKED.value;
      kernel_start       = plat_out.EX_CMD.KERNEL_START.value;
      kernel_done        = plat_out.EX_CMD.KERNEL_DONE.value;
      kernel_abort       = plat_out.EX_CMD.KERNEL_ABORT.value;
      bank_req_valid     = plat_out.EX_CMD.BANK_REQ.value;
      bank_req_tag       = plat_out.EX_TAG.TAG.value;
      bank_req_write     = plat_out.EX_ADDR.WRITE.value;
      bank_release_valid = plat_out.EX_CMD.RELEASE.value;
      bank_release_id    = plat_out.EX_ADDR.BANK.value;
      bank_flush_valid   = plat_out.EX_CMD.FLUSH.value;
      bank_flush_id      = plat_out.EX_ADDR.BANK.value;
      ld_valid           = plat_out.EX_CMD.LOAD.value;
      ld_bank            = plat_out.EX_ADDR.BANK.value;
      ld_row             = plat_out.EX_ADDR.ROW.value;
      st_valid           = plat_out.EX_CMD.STORE.value;
      st_bank            = plat_out.EX_ADDR.BANK.value;
      st_row             = plat_out.EX_ADDR.ROW.value;
      st_data            = ex_data;
      st_meta            = plat_out.EX_META.META.value;
    end else begin
      active_ctx         = ipu_active_ctx;
      ctx_locked         = ipu_ctx_locked;
      kernel_start       = ipu_kernel_start;
      kernel_done        = ipu_kernel_done;
      kernel_abort       = ipu_kernel_abort;
      bank_req_valid     = ipu_bank_req_valid;
      bank_req_tag       = ipu_bank_req_tag;
      bank_req_write     = ipu_bank_req_write;
      bank_release_valid = ipu_bank_release_valid;
      bank_release_id    = ipu_bank_release_id;
      bank_flush_valid   = ipu_bank_flush_valid;
      bank_flush_id      = ipu_bank_flush_id;
      ld_valid           = ipu_ld_valid;
      ld_bank            = ipu_ld_bank;
      ld_row             = ipu_ld_row;
      st_valid           = ipu_st_valid;
      st_bank            = ipu_st_bank;
      st_row             = ipu_st_row;
      st_data            = ipu_st_data;
      st_meta            = ipu_st_meta;
    end
  end

  // ---- The prototype cache unit -----------------------------------------------------

  table_cfg_t  cfg [2][TABLES];
  logic [1:0]  drained;
  logic        commit_err, config_err, fill_busy, drain_busy, dma_err;
  bank_state_e bank_state [XMEM_BANKS];
  logic [TAG_W-1:0] bank_tag [XMEM_BANKS];
  logic [XMEM_BANKS-1:0] bank_dirty;

  always_comb
    for (int k = 0; k < 2; k++)
      for (int t = 0; t < TABLES; t++) begin
        cfg[k][t].enable      = plat_out.TABLE_SET[k].TABLE[t].CFG.ENABLE.value;
        cfg[k][t].ttype       = table_type_e'(plat_out.TABLE_SET[k].TABLE[t].CFG.TYPE.value);
        cfg[k][t].format      = plat_out.TABLE_SET[k].TABLE[t].CFG.FORMAT.value;
        cfg[k][t].first_bank  = plat_out.TABLE_SET[k].TABLE[t].BANKS.FIRST_BANK.value;
        cfg[k][t].size        = plat_out.TABLE_SET[k].TABLE[t].SIZE.VALUE.value;
        cfg[k][t].jump_back   = plat_out.TABLE_SET[k].TABLE[t].JUMP_BACK.VALUE.value;
        cfg[k][t].repetitions = plat_out.TABLE_SET[k].TABLE[t].REPETITIONS.VALUE.value;
        cfg[k][t].dram        = {plat_out.TABLE_SET[k].TABLE[t].DRAM_HI.VALUE.value,
                                 plat_out.TABLE_SET[k].TABLE[t].DRAM_LO.VALUE.value};
      end

  proto_cache_unit #(
    .AXI_ADDR_W(C_M_AXI_HBM_ADDR_WIDTH),
    .AXI_ID_W  (C_M_AXI_HBM_ID_WIDTH)
  ) u_cu (
    .clk               (ap_clk),
    .rst_n             (cu_rst_n),
    .cfg               (cfg),
    .commit            ({plat_out.COMMIT.SET1.value, plat_out.COMMIT.SET0.value}),
    .tables_valid      (tables_valid),
    .drained           (drained),
    .commit_err        (commit_err),
    .config_err        (config_err),
    .bank_state        (bank_state),
    .bank_tag          (bank_tag),
    .bank_dirty        (bank_dirty),
    .fill_busy         (fill_busy),
    .drain_busy        (drain_busy),
    .dma_err           (dma_err),
    .tbl_view          (tbl_view),
    .active_ctx        (active_ctx),
    .ctx_locked        (ctx_locked),
    .kernel_done       (kernel_done),
    .kernel_abort      (kernel_abort),
    .bank_pub_valid    (bank_pub_valid),
    .bank_pub_id       (bank_pub_id),
    .bank_pub_tag      (bank_pub_tag),
    .bank_free_valid   (bank_free_valid),
    .bank_free_id      (bank_free_id),
    .bank_req_valid    (bank_req_valid),
    .bank_req_tag      (bank_req_tag),
    .bank_req_write    (bank_req_write),
    .bank_release_valid(bank_release_valid),
    .bank_release_id   (bank_release_id),
    .bank_flush_valid  (bank_flush_valid),
    .bank_flush_id     (bank_flush_id),
    .ld_valid          (ld_valid),
    .ld_bank           (ld_bank),
    .ld_row            (ld_row),
    .ld_data           (ld_data),
    .ld_meta           (ld_meta),
    .st_valid          (st_valid),
    .st_bank           (st_bank),
    .st_row            (st_row),
    .st_data           (st_data),
    .st_meta           (st_meta),
    .m_axi_awvalid     (m_axi_hbm_awvalid),
    .m_axi_awready     (m_axi_hbm_awready),
    .m_axi_awaddr      (m_axi_hbm_awaddr),
    .m_axi_awid        (m_axi_hbm_awid),
    .m_axi_awlen       (m_axi_hbm_awlen),
    .m_axi_awsize      (m_axi_hbm_awsize),
    .m_axi_awburst     (m_axi_hbm_awburst),
    .m_axi_wvalid      (m_axi_hbm_wvalid),
    .m_axi_wready      (m_axi_hbm_wready),
    .m_axi_wdata       (m_axi_hbm_wdata),
    .m_axi_wstrb       (m_axi_hbm_wstrb),
    .m_axi_wlast       (m_axi_hbm_wlast),
    .m_axi_bvalid      (m_axi_hbm_bvalid),
    .m_axi_bready      (m_axi_hbm_bready),
    .m_axi_bresp       (m_axi_hbm_bresp),
    .m_axi_bid         (m_axi_hbm_bid),
    .m_axi_arvalid     (m_axi_hbm_arvalid),
    .m_axi_arready     (m_axi_hbm_arready),
    .m_axi_araddr      (m_axi_hbm_araddr),
    .m_axi_arid        (m_axi_hbm_arid),
    .m_axi_arlen       (m_axi_hbm_arlen),
    .m_axi_arsize      (m_axi_hbm_arsize),
    .m_axi_arburst     (m_axi_hbm_arburst),
    .m_axi_rvalid      (m_axi_hbm_rvalid),
    .m_axi_rready      (m_axi_hbm_rready),
    .m_axi_rdata       (m_axi_hbm_rdata),
    .m_axi_rresp       (m_axi_hbm_rresp),
    .m_axi_rlast       (m_axi_hbm_rlast),
    .m_axi_rid         (m_axi_hbm_rid)
  );

  logic [BANK_W-1:0] last_pub_id, last_free_id;
  logic [TAG_W-1:0]  last_pub_tag;

  // ---- Platform register inputs ---------------------------------------------------

  always_comb begin
    plat_in = '{default: '0};
    plat_in.BUILD_ID.VALUE.next = BUILD_ID;
    plat_in.CLK_KHZ.VALUE.next  = CLK_KHZ;
    plat_in.STATUS.IPU_IRQ.next    = irq;
    plat_in.STATUS.FILL_BUSY.next  = fill_busy;
    plat_in.STATUS.DRAIN_BUSY.next = drain_busy;
    plat_in.STATUS.DMA_ERR.next    = dma_err;
    plat_in.CU_STATUS.TABLES_VALID.next = tables_valid;
    plat_in.CU_STATUS.DRAINED.next      = drained;
    plat_in.CU_STATUS.COMMIT_ERR.next   = commit_err;
    plat_in.CU_STATUS.CONFIG_ERR.next   = config_err;
    for (int b = 0; b < XMEM_BANKS; b++) begin
      plat_in.BANK[b].STATE.next = bank_state[b];
      plat_in.BANK[b].DIRTY.next = bank_dirty[b];
      plat_in.BANK[b].TAG.next   = bank_tag[b];
    end
    // The exerciser's load lands in EX_DATA and EX_META the cycle after LOAD.
    plat_in.EX_META.META.next = ld_meta;
    plat_in.EX_META.META.we   = ex_load_q;
    for (int i = 0; i < 32; i++) begin
      plat_in.EX_DATA[i].VALUE.next = ld_data[32*i +: 32];
      plat_in.EX_DATA[i].VALUE.we   = ex_load_q;
    end
    plat_in.EX_EVENTS.BANK_PUB.hwset  = bank_pub_valid;
    plat_in.EX_EVENTS.BANK_FREE.hwset = bank_free_valid;
    plat_in.EX_LAST.PUB_ID.next  = last_pub_id;
    plat_in.EX_LAST.PUB_TAG.next = last_pub_tag;
    plat_in.EX_LAST.FREE_ID.next = last_free_id;
  end

  always_ff @(posedge ap_clk or negedge plat_rst_n) begin
    if (!plat_rst_n) begin
      last_pub_id  <= '0;
      last_pub_tag <= '0;
      last_free_id <= '0;
    end else begin
      if (bank_pub_valid) begin
        last_pub_id  <= bank_pub_id;
        last_pub_tag <= bank_pub_tag;
      end
      if (bank_free_valid) last_free_id <= bank_free_id;
    end
  end

  logic unused;
  assign unused = ^{kernel_start, ipu_bid, ipu_rid, ipu_rlast};

endmodule
