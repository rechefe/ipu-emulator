// Types of the prototype platform (fpga-prototype.md).
package proto_pkg;
  import ipu_socket_pkg::*;

  // One table as the host configures it (proto_platform.rdl, TABLE_SET[k].TABLE[t]).
  typedef struct packed {
    logic        enable;
    table_type_e ttype;
    logic [8:0]  format;
    logic [BANK_W-1:0] first_bank;
    logic [15:0] size;
    logic [15:0] jump_back;
    logic [15:0] repetitions;
    logic [63:0] dram;
  } table_cfg_t;

  typedef enum logic [1:0] {
    BANK_FREE     = 2'd0,
    BANK_FILLING  = 2'd1,
    BANK_VALID    = 2'd2,
    BANK_DRAINING = 2'd3
  } bank_state_e;

  // DMA: 512-bit beats, 64-beat bursts (4 KiB, the AXI boundary), 32 bursts a bank.
  localparam int DMA_DATA_W = 512;
  localparam int BEATS_PER_ROW = ROW_BITS / DMA_DATA_W;
  localparam int BURST_BEATS = 64;
  localparam int ROWS_PER_BURST = BURST_BEATS / BEATS_PER_ROW;
  localparam int BURSTS_PER_BANK = XMEM_ROWS / ROWS_PER_BURST;
  localparam int BANK_BYTES = XMEM_ROWS * ROW_BITS / 8;
  localparam int BURST_BYTES = BURST_BEATS * DMA_DATA_W / 8;
endpackage
