// One AXI4-Lite slave port to two, by one address bit. One write and one read
// in flight at a time: enough for a host that issues one access at a time
// (XRT register reads and writes). The address and data buses go to both
// masters unchanged; only the handshakes are routed.
module axil_split2 #(
  parameter int unsigned ADDR_W = 18,
  parameter int unsigned SEL_BIT = 17
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              s_awvalid,
  output logic              s_awready,
  input  logic [ADDR_W-1:0] s_awaddr,
  input  logic              s_wvalid,
  output logic              s_wready,
  output logic              s_bvalid,
  input  logic              s_bready,
  output logic [1:0]        s_bresp,
  input  logic              s_arvalid,
  output logic              s_arready,
  input  logic [ADDR_W-1:0] s_araddr,
  output logic              s_rvalid,
  input  logic              s_rready,
  output logic [31:0]       s_rdata,
  output logic [1:0]        s_rresp,

  output logic              m0_awvalid, m1_awvalid,
  input  logic              m0_awready, m1_awready,
  output logic              m0_wvalid,  m1_wvalid,
  input  logic              m0_wready,  m1_wready,
  input  logic              m0_bvalid,  m1_bvalid,
  output logic              m0_bready,  m1_bready,
  input  logic [1:0]        m0_bresp,   m1_bresp,
  output logic              m0_arvalid, m1_arvalid,
  input  logic              m0_arready, m1_arready,
  input  logic              m0_rvalid,  m1_rvalid,
  output logic              m0_rready,  m1_rready,
  input  logic [31:0]       m0_rdata,   m1_rdata,
  input  logic [1:0]        m0_rresp,   m1_rresp
);

  // ---- Writes: wait for AW and W together, pass both, then pass B. ----
  typedef enum logic [1:0] {W_IDLE, W_ADDR_DATA, W_RESP} wstate_e;
  wstate_e wstate;
  logic    wsel, aw_done, w_done;
  logic    awready, wready, bvalid;

  assign awready = wsel ? m1_awready : m0_awready;
  assign wready  = wsel ? m1_wready  : m0_wready;
  assign bvalid  = wsel ? m1_bvalid  : m0_bvalid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wstate  <= W_IDLE;
      wsel    <= 1'b0;
      aw_done <= 1'b0;
      w_done  <= 1'b0;
    end else begin
      case (wstate)
        W_IDLE: if (s_awvalid && s_wvalid) begin
          wsel    <= s_awaddr[SEL_BIT];
          aw_done <= 1'b0;
          w_done  <= 1'b0;
          wstate  <= W_ADDR_DATA;
        end
        W_ADDR_DATA: begin
          if (awready) aw_done <= 1'b1;
          if (wready) w_done <= 1'b1;
          if ((aw_done || awready) && (w_done || wready)) wstate <= W_RESP;
        end
        W_RESP: if (bvalid && s_bready) wstate <= W_IDLE;
        default: wstate <= W_IDLE;
      endcase
    end
  end

  logic aw_go, w_go;
  assign aw_go = wstate == W_ADDR_DATA && !aw_done;
  assign w_go  = wstate == W_ADDR_DATA && !w_done;
  assign m0_awvalid = aw_go && !wsel;
  assign m1_awvalid = aw_go && wsel;
  assign m0_wvalid  = w_go && !wsel;
  assign m1_wvalid  = w_go && wsel;
  assign s_awready  = aw_go && awready;
  assign s_wready   = w_go && wready;
  assign s_bvalid   = wstate == W_RESP && bvalid;
  assign s_bresp    = wsel ? m1_bresp : m0_bresp;
  assign m0_bready  = wstate == W_RESP && !wsel && s_bready;
  assign m1_bready  = wstate == W_RESP && wsel && s_bready;

  // ---- Reads: pass AR, then pass R. ----
  typedef enum logic [1:0] {R_IDLE, R_ADDR, R_DATA} rstate_e;
  rstate_e rstate;
  logic    rsel, arready, rvalid;

  assign arready = rsel ? m1_arready : m0_arready;
  assign rvalid  = rsel ? m1_rvalid  : m0_rvalid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rstate <= R_IDLE;
      rsel   <= 1'b0;
    end else begin
      case (rstate)
        R_IDLE: if (s_arvalid) begin
          rsel   <= s_araddr[SEL_BIT];
          rstate <= R_ADDR;
        end
        R_ADDR: if (arready) rstate <= R_DATA;
        R_DATA: if (rvalid && s_rready) rstate <= R_IDLE;
        default: rstate <= R_IDLE;
      endcase
    end
  end

  assign m0_arvalid = rstate == R_ADDR && !rsel;
  assign m1_arvalid = rstate == R_ADDR && rsel;
  assign s_arready  = rstate == R_ADDR && arready;
  assign s_rvalid   = rstate == R_DATA && rvalid;
  assign s_rdata    = rsel ? m1_rdata : m0_rdata;
  assign s_rresp    = rsel ? m1_rresp : m0_rresp;
  assign m0_rready  = rstate == R_DATA && !rsel && s_rready;
  assign m1_rready  = rstate == R_DATA && rsel && s_rready;

endmodule
