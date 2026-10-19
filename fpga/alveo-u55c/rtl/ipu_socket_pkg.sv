// The IPU socket: the sizes and types on the ports between `ipu_top` and the
// platform (docs/content/requirements/interfaces.md). The RTL repository's
// `ipu_top` imports this package; the platform builds against it.
//
// Status: prototype default. OQ-15 leaves `tbl_view` and the AXI ID width
// open; the values here are the prototype's (fpga-prototype.md, "The socket").
package ipu_socket_pkg;

  // XMEM (cache-unit.md, section 4).
  localparam int XMEM_BANKS = 16;
  localparam int XMEM_ROWS = 1024;
  localparam int ROW_BITS = 1024;
  localparam int META_BITS = 16;
  localparam int BANK_W = $clog2(XMEM_BANKS);
  localparam int ROW_W = $clog2(XMEM_ROWS);

  // Tables and tags: a tag is {ctx, table_id[3:0], page[9:0]} (interfaces.md).
  localparam int TABLES = 16;
  localparam int TABLE_W = $clog2(TABLES);
  localparam int PAGE_W = 10;
  localparam int TAG_W = 1 + TABLE_W + PAGE_W;

  // Host port: AXI4, 32-bit data (IPU-HOST-1). The address width is the
  // register map's (ipu_host_addr_pkg::ADDR_WIDTH).
  localparam int HOST_DATA_W = 32;
  localparam int HOST_ID_W = 4;

  typedef enum logic [1:0] {
    TABLE_READ_ONLY        = 2'd0,
    TABLE_WRITE_ONLY       = 2'd1,
    TABLE_READ_AFTER_WRITE = 2'd2,
    TABLE_SCRATCH_PAD      = 2'd3
  } table_type_e;

  // What the cache agent needs of one table: F-6 (size), F-7 (type), and the
  // release and flush rule (jump_back, repetitions; cache-unit.md 9.1, 9.2).
  typedef struct packed {
    logic        enable;
    table_type_e ttype;
    logic [8:0]  format;
    logic [15:0] size;         // pages
    logic [15:0] jump_back;    // rows; 0: no release or flush before kernel done
    logic [15:0] repetitions;
  } table_view_t;

  typedef table_view_t [TABLES-1:0] table_set_view_t;

  // tbl_view: both table sets, as committed (CU-3: stable while locked).
  typedef table_set_view_t [1:0] tbl_view_t;
  localparam int TBL_W = $bits(tbl_view_t);

  function automatic logic [TAG_W-1:0] make_tag(logic ctx, logic [TABLE_W-1:0] table_id,
                                                logic [PAGE_W-1:0] page);
    return {ctx, table_id, page};
  endfunction

endpackage
