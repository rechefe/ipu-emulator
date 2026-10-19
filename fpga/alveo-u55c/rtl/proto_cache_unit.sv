// The prototype cache unit: XMEM, the bank states, and a DMA to device memory
// (fpga-prototype.md, "The prototype cache unit").
//
// Fixed mapping: page p of table t in table set k lives in bank
// first_bank(t) + p, and no page is ever replaced. At a commit the unit plans,
// for every bank, which (set, table, page) it holds; it then fills the read
// banks from DRAM and claims the write-only ones as each bank becomes free.
// The IPU's events move banks on (interfaces.md, Bank events and requests):
//   bank_release: VALID -> FREE         (CU-25)
//   bank_flush:   VALID -> DRAINING -> FREE, then bank_free (CU-24)
//   bank_req:     re-plan a released bank of a committed set (CU-22, CU-23)
//   kernel_done:  the kernel's written banks drain, the rest free (CU-17)
//   kernel_abort: every bank of the kernel frees, none drains (OQ-12 proposal)
// A bank leaves VALID only on those events (CU-20), and is published only
// after its whole fill (CU-21).
module proto_cache_unit
  import ipu_socket_pkg::*;
  import proto_pkg::*;
#(
  parameter int unsigned AXI_ADDR_W = 64,
  parameter int unsigned AXI_ID_W = 1
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Configuration and status (proto_platform registers)
  input  table_cfg_t              cfg [2][TABLES],
  input  logic [1:0]              commit,          // one-cycle pulse per set
  output logic [1:0]              tables_valid,
  output logic [1:0]              drained,
  output logic                    commit_err,
  output logic                    config_err,
  output bank_state_e             bank_state [XMEM_BANKS],
  output logic [TAG_W-1:0]        bank_tag [XMEM_BANKS],
  output logic [XMEM_BANKS-1:0]   bank_dirty,
  output logic                    fill_busy,
  output logic                    drain_busy,
  output logic                    dma_err,

  // IPU side
  output tbl_view_t               tbl_view,
  input  logic                    active_ctx,
  input  logic [1:0]              ctx_locked,
  input  logic                    kernel_done,
  input  logic                    kernel_abort,
  output logic                    bank_pub_valid,
  output logic [BANK_W-1:0]       bank_pub_id,
  output logic [TAG_W-1:0]        bank_pub_tag,
  output logic                    bank_free_valid,
  output logic [BANK_W-1:0]       bank_free_id,
  input  logic                    bank_req_valid,
  input  logic [TAG_W-1:0]        bank_req_tag,
  input  logic                    bank_req_write,
  input  logic                    bank_release_valid,
  input  logic [BANK_W-1:0]       bank_release_id,
  input  logic                    bank_flush_valid,
  input  logic [BANK_W-1:0]       bank_flush_id,
  input  logic                    ld_valid,
  input  logic [BANK_W-1:0]       ld_bank,
  input  logic [ROW_W-1:0]        ld_row,
  output logic [ROW_BITS-1:0]     ld_data,
  output logic [META_BITS-1:0]    ld_meta,
  input  logic                    st_valid,
  input  logic [BANK_W-1:0]       st_bank,
  input  logic [ROW_W-1:0]        st_row,
  input  logic [ROW_BITS-1:0]     st_data,
  input  logic [META_BITS-1:0]    st_meta,

  // AXI4 master to device memory (HBM)
  output logic                    m_axi_awvalid,
  input  logic                    m_axi_awready,
  output logic [AXI_ADDR_W-1:0]   m_axi_awaddr,
  output logic [AXI_ID_W-1:0]     m_axi_awid,
  output logic [7:0]              m_axi_awlen,
  output logic [2:0]              m_axi_awsize,
  output logic [1:0]              m_axi_awburst,
  output logic                    m_axi_wvalid,
  input  logic                    m_axi_wready,
  output logic [DMA_DATA_W-1:0]   m_axi_wdata,
  output logic [DMA_DATA_W/8-1:0] m_axi_wstrb,
  output logic                    m_axi_wlast,
  input  logic                    m_axi_bvalid,
  output logic                    m_axi_bready,
  input  logic [1:0]              m_axi_bresp,
  input  logic [AXI_ID_W-1:0]     m_axi_bid,
  output logic                    m_axi_arvalid,
  input  logic                    m_axi_arready,
  output logic [AXI_ADDR_W-1:0]   m_axi_araddr,
  output logic [AXI_ID_W-1:0]     m_axi_arid,
  output logic [7:0]              m_axi_arlen,
  output logic [2:0]              m_axi_arsize,
  output logic [1:0]              m_axi_arburst,
  input  logic                    m_axi_rvalid,
  output logic                    m_axi_rready,
  input  logic [DMA_DATA_W-1:0]   m_axi_rdata,
  input  logic [1:0]              m_axi_rresp,
  input  logic                    m_axi_rlast,
  input  logic [AXI_ID_W-1:0]     m_axi_rid
);

  localparam int WORD_W = ROW_BITS + META_BITS;
  localparam int BURST_W = $clog2(BURSTS_PER_BANK);
  localparam int BEAT_W = $clog2(BURST_BEATS);

  // The DMA counters wrap exactly at a burst's last beat and a bank's last burst.
  if (BURST_BEATS != 2 ** BEAT_W || BURSTS_PER_BANK != 2 ** BURST_W ||
      BANK_BYTES != 2 ** 17 || BURST_BYTES != 2 ** 12) begin : g_geometry
    $error("proto_cache_unit: the DMA address arithmetic assumes 64-beat bursts, 4 KiB and 128 KiB");
  end

  // ---- Committed table sets and the bank plan ---------------------------------

  table_cfg_t wc [2][TABLES];  // working copies, taken at commit

  // What bank b holds for set k: valid, table, page, and whether it is filled
  // from DRAM (any type but write-only) or only claimed.
  logic                 plan_valid [2][XMEM_BANKS];
  logic [TABLE_W-1:0]   plan_table [2][XMEM_BANKS];
  logic [BANK_W-1:0]    plan_page  [2][XMEM_BANKS];
  logic                 plan_fill  [2][XMEM_BANKS];
  logic [XMEM_BANKS-1:0] want [2];   // bank still to fill or claim for set k
  logic [1:0]           done_seen;   // set k's last kernel is done or aborted

  // The plan a commit of `c` would make, and whether it is consistent: every
  // enabled table fits in the 16 banks and no two claim the same bank.
  logic                 new_valid [XMEM_BANKS];
  logic [TABLE_W-1:0]   new_table [XMEM_BANKS];
  logic [BANK_W-1:0]    new_page  [XMEM_BANKS];
  logic                 new_fill  [XMEM_BANKS];
  logic                 new_bad;

  function automatic void plan_set(input table_cfg_t tables [TABLES],
                                   output logic valid [XMEM_BANKS],
                                   output logic [TABLE_W-1:0] tbl [XMEM_BANKS],
                                   output logic [BANK_W-1:0] page [XMEM_BANKS],
                                   output logic fill [XMEM_BANKS],
                                   output logic bad);
    bad = 1'b0;
    for (int b = 0; b < XMEM_BANKS; b++) begin
      valid[b] = 1'b0;
      tbl[b] = '0;
      page[b] = '0;
      fill[b] = 1'b0;
    end
    for (int t = 0; t < TABLES; t++) begin
      if (tables[t].enable) begin
        if (32'(tables[t].first_bank) + 32'(tables[t].size) > unsigned'(XMEM_BANKS)) bad = 1'b1;
        for (int b = 0; b < XMEM_BANKS; b++) begin
          if (b >= int'(tables[t].first_bank) && b < int'(tables[t].first_bank) + int'(tables[t].size)) begin
            if (valid[b]) bad = 1'b1;
            valid[b] = 1'b1;
            tbl[b] = TABLE_W'(t);
            page[b] = BANK_W'(b - int'(tables[t].first_bank));
            fill[b] = tables[t].ttype != TABLE_WRITE_ONLY;
          end
        end
      end
    end
  endfunction

  logic       commit_ctx;
  logic       commit_any;
  always_comb begin
    commit_any = |commit;
    commit_ctx = commit[1];  // a write commits one set; set 1 wins a double write
    plan_set(cfg[commit_ctx], new_valid, new_table, new_page, new_fill, new_bad);
  end

  // Banks that still hold something of set k.
  logic [1:0] ctx_busy;
  always_comb begin
    ctx_busy = '0;
    for (int b = 0; b < XMEM_BANKS; b++)
      if (bank_state[b] != BANK_FREE) ctx_busy[bank_tag[b][TAG_W-1]] = 1'b1;
  end

  logic commit_ok;
  assign commit_ok = commit_any && !ctx_locked[commit_ctx] && !tables_valid[commit_ctx] &&
                     !ctx_busy[commit_ctx] && !new_bad;

  // ---- XMEM --------------------------------------------------------------------

  logic                 dma_rd_en;        // the DMA reads a row (drain)
  logic [ROW_W-1:0]     dma_row;
  logic [BANK_W-1:0]    dma_bank;
  logic                 dma_wr_en;        // the DMA writes a row (fill)
  logic [ROW_BITS-1:0]  dma_wr_data;
  logic [WORD_W-1:0]    rd_q [XMEM_BANKS];
  logic [BANK_W-1:0]    ld_bank_q;

  for (genvar b = 0; b < XMEM_BANKS; b++) begin : g_bank
    (* ram_style = "ultra" *) logic [WORD_W-1:0] mem [XMEM_ROWS];
    logic [ROW_W-1:0] rd_addr;
    // The IPU never loads from a bank the DMA drains (it is not VALID).
    assign rd_addr = (dma_rd_en && dma_bank == BANK_W'(unsigned'(b))) ? dma_row : ld_row;
    always_ff @(posedge clk) begin
      rd_q[b] <= mem[rd_addr];
      // The IPU stores only to VALID banks; the DMA fills only FILLING ones.
      if (st_valid && st_bank == BANK_W'(unsigned'(b)))
        mem[st_row] <= {st_meta, st_data};
      else if (dma_wr_en && dma_bank == BANK_W'(unsigned'(b)))
        mem[dma_row] <= {META_BITS'(0), dma_wr_data};
    end
  end

  // CU-1: the row arrives one cycle after ld_valid.
  always_ff @(posedge clk) if (ld_valid) ld_bank_q <= ld_bank;
  assign ld_data = rd_q[ld_bank_q][ROW_BITS-1:0];
  assign ld_meta = rd_q[ld_bank_q][WORD_W-1:ROW_BITS];

  // ---- DMA -----------------------------------------------------------------------

  typedef enum logic [3:0] {
    DMA_IDLE, FILL_AR, FILL_R, FILL_DONE, DRAIN_AW, DRAIN_RD, DRAIN_LAT, DRAIN_W0, DRAIN_W1,
    DRAIN_B, DRAIN_DONE
  } dma_state_e;

  dma_state_e         dma_state;
  logic [BURST_W-1:0] burst;
  logic [BEAT_W-1:0]  beat;
  logic [63:0]        dma_base;
  logic [DMA_DATA_W-1:0] fill_lo;
  logic [ROW_BITS-1:0]   drain_row;
  logic [XMEM_BANKS-1:0] drain_pending;
  logic [XMEM_BANKS-1:0] cancel;     // a fill whose kernel ended before it finished

  // The device address of page 0 of the table a tag names.
  function automatic logic [63:0] page_address(input logic [TAG_W-1:0] t);
    table_cfg_t c;
    c = wc[t[TAG_W-1]][t[PAGE_W +: TABLE_W]];
    return c.dram + {37'h0, t[PAGE_W-1:0], 17'h0};  // page * 128 KiB (BANK_BYTES)
  endfunction

  // The scheduler's choice, made while the DMA is idle: a drain first, then a
  // fill or claim for the active set, then for the other set.
  logic                 pick_valid, pick_drain, pick_ctx;
  logic [BANK_W-1:0]    pick_bank;
  always_comb begin
    pick_valid = 1'b0;
    pick_drain = 1'b0;
    pick_ctx = active_ctx;
    pick_bank = '0;
    for (int b = XMEM_BANKS - 1; b >= 0; b--)
      if (drain_pending[b]) begin
        pick_valid = 1'b1;
        pick_drain = 1'b1;
        pick_bank = BANK_W'(b);
      end
    for (int k = 1; k >= 0; k--) begin
      logic ctx;
      ctx = (k == 0) ? active_ctx : !active_ctx;
      if (!pick_valid && tables_valid[ctx])
        for (int b = XMEM_BANKS - 1; b >= 0; b--)
          if (want[ctx][b] && bank_state[b] == BANK_FREE) begin
            pick_valid = 1'b1;
            pick_ctx = ctx;
            pick_bank = BANK_W'(b);
          end
    end
  end

  assign dma_rd_en   = dma_state == DRAIN_RD;
  assign dma_row     = {burst, beat[BEAT_W-1:1]};  // the row of the current beat pair
  assign dma_wr_en   = dma_state == FILL_R && m_axi_rvalid && beat[0];
  assign dma_wr_data = {m_axi_rdata, fill_lo};

  assign m_axi_arvalid = dma_state == FILL_AR;
  assign m_axi_araddr  = dma_base + {47'h0, burst, 12'h000};  // burst * 4 KiB (BURST_BYTES)
  assign m_axi_arid    = '0;
  assign m_axi_arlen   = 8'(BURST_BEATS - 1);
  assign m_axi_arsize  = 3'($clog2(DMA_DATA_W / 8));
  assign m_axi_arburst = 2'b01;
  assign m_axi_rready  = dma_state == FILL_R;

  assign m_axi_awvalid = dma_state == DRAIN_AW;
  assign m_axi_awaddr  = m_axi_araddr;
  assign m_axi_awid    = '0;
  assign m_axi_awlen   = 8'(BURST_BEATS - 1);
  assign m_axi_awsize  = 3'($clog2(DMA_DATA_W / 8));
  assign m_axi_awburst = 2'b01;
  assign m_axi_wvalid  = dma_state == DRAIN_W0 || dma_state == DRAIN_W1;
  assign m_axi_wdata   = (dma_state == DRAIN_W1) ? drain_row[ROW_BITS-1:DMA_DATA_W]
                                                 : drain_row[DMA_DATA_W-1:0];
  assign m_axi_wstrb   = '1;
  assign m_axi_wlast   = dma_state == DRAIN_W1 && beat == '1;
  assign m_axi_bready  = dma_state == DRAIN_B;

  assign fill_busy  = dma_state inside {FILL_AR, FILL_R, FILL_DONE};
  assign drain_busy = dma_state inside {DRAIN_AW, DRAIN_RD, DRAIN_LAT, DRAIN_W0, DRAIN_W1,
                                        DRAIN_B, DRAIN_DONE};

  // ---- Bank states and events ---------------------------------------------------

  logic done_ctx;       // the context of a kernel_done: active_ctx before a handoff flip
  logic active_ctx_q;
  assign done_ctx = active_ctx_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int b = 0; b < XMEM_BANKS; b++) begin
        bank_state[b] <= BANK_FREE;   // CU-4
        bank_tag[b]   <= '0;
        for (int k = 0; k < 2; k++) begin
          plan_valid[k][b] <= 1'b0;
          plan_table[k][b] <= '0;
          plan_page[k][b]  <= '0;
          plan_fill[k][b]  <= 1'b0;
        end
      end
      for (int k = 0; k < 2; k++)
        for (int t = 0; t < TABLES; t++) wc[k][t] <= '0;
      want           <= '{default: '0};
      bank_dirty     <= '0;
      drain_pending  <= '0;
      cancel         <= '0;
      tables_valid   <= '0;            // CU-4
      done_seen      <= '1;
      commit_err     <= 1'b0;
      config_err     <= 1'b0;
      dma_err        <= 1'b0;
      dma_state      <= DMA_IDLE;
      dma_bank       <= '0;
      dma_base       <= '0;
      burst          <= '0;
      beat           <= '0;
      fill_lo        <= '0;
      drain_row      <= '0;
      bank_pub_valid <= 1'b0;
      bank_pub_id    <= '0;
      bank_pub_tag   <= '0;
      bank_free_valid <= 1'b0;
      bank_free_id   <= '0;
      active_ctx_q   <= 1'b0;
    end else begin
      bank_pub_valid  <= 1'b0;
      bank_free_valid <= 1'b0;
      active_ctx_q    <= active_ctx;

      // Commit (CU-11, CU-13, CU-14).
      if (commit_any) begin
        commit_err <= !commit_ok;
        config_err <= new_bad;
        if (commit_ok) begin
          wc[commit_ctx] <= cfg[commit_ctx];
          for (int b = 0; b < XMEM_BANKS; b++) begin
            plan_valid[commit_ctx][b] <= new_valid[b];
            plan_table[commit_ctx][b] <= new_table[b];
            plan_page[commit_ctx][b]  <= new_page[b];
            plan_fill[commit_ctx][b]  <= new_fill[b];
            want[commit_ctx][b]       <= new_valid[b];
          end
          tables_valid[commit_ctx] <= 1'b1;
          done_seen[commit_ctx]    <= 1'b0;
        end
      end

      // bank_req for a planned page that is not held: plan it again (CU-22).
      // A tag already filling or valid is ignored (CU-23).
      if (bank_req_valid && tables_valid[bank_req_tag[TAG_W-1]]) begin
        for (int b = 0; b < XMEM_BANKS; b++) begin
          logic k;
          k = bank_req_tag[TAG_W-1];
          if (plan_valid[k][b] && plan_table[k][b] == bank_req_tag[PAGE_W +: TABLE_W] &&
              PAGE_W'(plan_page[k][b]) == bank_req_tag[PAGE_W-1:0] &&
              !(bank_state[b] inside {BANK_FILLING, BANK_VALID} && bank_tag[b] == bank_req_tag))
            want[k][b] <= 1'b1;
        end
      end

      if (st_valid) bank_dirty[st_bank] <= 1'b1;

      if (bank_release_valid && bank_state[bank_release_id] == BANK_VALID) begin
        bank_state[bank_release_id] <= BANK_FREE;    // CU-25
        bank_dirty[bank_release_id] <= 1'b0;
      end
      if (bank_flush_valid && bank_state[bank_flush_id] == BANK_VALID) begin
        bank_state[bank_flush_id] <= BANK_DRAINING;  // CU-24
        drain_pending[bank_flush_id] <= 1'b1;
      end

      // Kernel done (CU-12, CU-17) and abort.
      if (kernel_done || kernel_abort) begin
        logic k;
        k = kernel_done ? done_ctx : active_ctx;
        for (int b = 0; b < XMEM_BANKS; b++) begin
          if (bank_tag[b][TAG_W-1] == k) begin
            if (bank_state[b] == BANK_VALID) begin
              if (kernel_done && bank_dirty[b]) begin
                bank_state[b] <= BANK_DRAINING;
                drain_pending[b] <= 1'b1;
              end else begin
                bank_state[b] <= BANK_FREE;
                bank_dirty[b] <= 1'b0;
              end
            end else if (bank_state[b] == BANK_FILLING) begin
              cancel[b] <= 1'b1;
            end
          end
        end
        want[k]         <= '0;
        tables_valid[k] <= 1'b0;
        done_seen[k]    <= 1'b1;
      end

      // DMA.
      case (dma_state)
        DMA_IDLE: begin
          burst <= '0;
          beat  <= '0;
          if (pick_valid && pick_drain) begin
            dma_bank <= pick_bank;
            dma_base <= page_address(bank_tag[pick_bank]);
            drain_pending[pick_bank] <= 1'b0;
            dma_state <= DRAIN_AW;
          end else if (pick_valid) begin
            logic [TAG_W-1:0] t;
            t = make_tag(pick_ctx, plan_table[pick_ctx][pick_bank],
                         PAGE_W'(plan_page[pick_ctx][pick_bank]));
            want[pick_ctx][pick_bank] <= 1'b0;
            bank_tag[pick_bank] <= t;
            bank_dirty[pick_bank] <= 1'b0;
            if (plan_fill[pick_ctx][pick_bank]) begin
              bank_state[pick_bank] <= BANK_FILLING;
              cancel[pick_bank] <= 1'b0;
              dma_bank <= pick_bank;
              dma_base <= page_address(t);
              dma_state <= FILL_AR;
            end else begin
              bank_state[pick_bank] <= BANK_VALID;   // a write-only page: claimed, not filled
              bank_pub_valid <= 1'b1;
              bank_pub_id    <= pick_bank;
              bank_pub_tag   <= t;
            end
          end
        end
        FILL_AR: if (m_axi_arready) dma_state <= FILL_R;
        FILL_R: if (m_axi_rvalid) begin
          if (m_axi_rresp != 2'b00) dma_err <= 1'b1;
          if (!beat[0]) fill_lo <= m_axi_rdata;
          beat <= beat + 1'b1;
          if (m_axi_rlast) begin
            burst <= burst + 1'b1;
            dma_state <= (burst == '1) ? FILL_DONE : FILL_AR;
          end
        end
        FILL_DONE: begin
          // CU-21: publish only a whole bank, and only for a set still committed.
          if (cancel[dma_bank] || !tables_valid[bank_tag[dma_bank][TAG_W-1]]) begin
            bank_state[dma_bank] <= BANK_FREE;
          end else begin
            bank_state[dma_bank] <= BANK_VALID;
            bank_pub_valid <= 1'b1;
            bank_pub_id    <= dma_bank;
            bank_pub_tag   <= bank_tag[dma_bank];
          end
          cancel[dma_bank] <= 1'b0;
          dma_state <= DMA_IDLE;
        end
        DRAIN_AW: if (m_axi_awready) dma_state <= DRAIN_RD;
        DRAIN_RD: dma_state <= DRAIN_LAT;
        DRAIN_LAT: begin
          drain_row <= rd_q[dma_bank][ROW_BITS-1:0];
          dma_state <= DRAIN_W0;
        end
        DRAIN_W0: if (m_axi_wready) begin
          beat <= beat + 1'b1;
          dma_state <= DRAIN_W1;
        end
        DRAIN_W1: if (m_axi_wready) begin
          beat <= beat + 1'b1;
          dma_state <= (beat == '1) ? DRAIN_B : DRAIN_RD;
        end
        DRAIN_B: if (m_axi_bvalid) begin
          if (m_axi_bresp != 2'b00) dma_err <= 1'b1;
          burst <= burst + 1'b1;
          dma_state <= (burst == '1) ? DRAIN_DONE : DRAIN_AW;
        end
        DRAIN_DONE: begin
          bank_state[dma_bank] <= BANK_FREE;
          bank_dirty[dma_bank] <= 1'b0;
          bank_free_valid <= 1'b1;
          bank_free_id    <= dma_bank;
          dma_state <= DMA_IDLE;
        end
        default: dma_state <= DMA_IDLE;
      endcase
    end
  end

  // DRAINED[k] (CU-18): set k's last kernel ended and none of its banks is left.
  always_comb
    for (int k = 0; k < 2; k++) drained[k] = done_seen[k] && !ctx_busy[k];

  // tbl_view (CU-3: the working copy changes only at a commit, which a lock refuses).
  always_comb
    for (int k = 0; k < 2; k++)
      for (int t = 0; t < TABLES; t++) begin
        tbl_view[k][t].enable      = wc[k][t].enable;
        tbl_view[k][t].ttype       = wc[k][t].ttype;
        tbl_view[k][t].format      = wc[k][t].format;
        tbl_view[k][t].size        = wc[k][t].size;
        tbl_view[k][t].jump_back   = wc[k][t].jump_back;
        tbl_view[k][t].repetitions = wc[k][t].repetitions;
      end

  logic unused;
  assign unused = ^{bank_req_write, m_axi_bid, m_axi_rid};

endmodule
