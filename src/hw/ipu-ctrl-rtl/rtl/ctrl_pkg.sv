// ctrl_pkg.sv — shared parameters and types for the CTRL-stage RTL.
//
// Every constant here is derived from a specific Python source in
// src/tools/ipu-common and src/tools/ipu-emu-py, cited per constant. Do not
// hand-tune a value to make a testbench pass — if a number here stops
// matching its cited source, the source moved and this file is stale, not
// the other way around. See planning/ctrl-stage-rtl/ for the epic this
// belongs to and which issue owns which part of this package.
package ctrl_pkg;

  // -------------------------------------------------------------------
  // Register file geometry
  //   ipu_common/ipu_config.py: LR_CR_SCALAR_BITS = 20
  //   ipu_common/registers.py:  REGISTER_DEFINITIONS["lr"]["count"] = 16
  //                             REGISTER_DEFINITIONS["cr"]["count"] = 16
  // -------------------------------------------------------------------
  localparam int unsigned REG_WIDTH  = 20;
  localparam int unsigned LR_COUNT   = 16;
  localparam int unsigned CR_COUNT   = 16;  // CR0-CR15; the CrIdx *operand type*
                                             // only ever encodes CR0-CR14 (CR15
                                             // is reserved for dstructure) — see
                                             // LCR_IDX_W note below.
  localparam int unsigned LR_IDX_W   = 4;   // $clog2(LR_COUNT)
  localparam int unsigned CR_IDX_W   = 4;   // $clog2(CR_COUNT), also CrIdx operand width
  localparam int unsigned LCR_IDX_W  = 5;   // LcrIdx: 0..15 = LR0-15, 16..30 = CR0-14
                                             // (lr_count + cr_count - 1).bit_length() = 5
                                             // -- see union_layout.get_operand_type_bits()

  // -------------------------------------------------------------------
  // LR-slot union layout.
  //
  // Ground truth: ipu_common.union_layout.compute_slot_layout("lr", ...),
  // confirmed by evaluating SLOT_UNIONS["lr"] directly:
  //
  //   field0 (5b, canonical LcrIdx): SET.src (CrIdx!) | ADD/SUB.src_b,
  //                                  INCR_MOD_POW2.step (LcrIdx) | INC/DEC.imm
  //   field1 (4b, LrIdx):            dest/reg for every non-NOP opcode
  //   field2 (5b, LrIdx/K):          ADD/SUB.src_a (LrIdx, 4 bits used) |
  //                                  INCR_MOD_POW2.k (4 bits used, 1 pad bit)
  //
  // Natural width is opcode(3) + 5 + 4 + 4 = 16 bits; union_layout.py pads
  // the LR slot to a fixed 17-bit target (_SLOT_TARGET_BITS["lr"] = 17,
  // "keeps encoded LR sub-instructions stable"), and the solver puts the
  // single padding bit into field2 (the field carrying the k operand) --
  // that bit is always 0 and unused by every opcode. This is why field2
  // is listed as 5 bits below even though no opcode needs more than 4.
  //
  // *** IMPORTANT — SET has NO immediate mode. ***
  // stage-control.md section 10.1.1 describes SET's operand as a 5-bit
  // "src5" with a mode-select MSB choosing between a CR read and a signed
  // 4-bit immediate. That is not what's implemented: SET's operand type in
  // instruction_spec.py is plain "CrIdx" (a bare 0-14 config-register
  // index), and execute_lr_set() in ipu.py has no immediate branch at all
  // -- it is unconditionally `reg = CR[src]`. The assembler's own generated
  // docs (ipu_as/gen_docs.py) agree: "SET copies from a cr register."
  // This RTL implements the real (CR-read-only) behavior. See
  // planning/ctrl-stage-rtl/01-spec-emulator-reconciliation.md — this
  // needs a stage-control.md correction, not a workaround here.
  // -------------------------------------------------------------------
  localparam int unsigned LR_OPCODE_W = 3;   // 7 opcodes -> (7-1).bit_length() = 3
  localparam int unsigned LR_FIELD0_W = 5;
  localparam int unsigned LR_FIELD1_W = 4;
  localparam int unsigned LR_FIELD2_W = 5;   // 4 meaningful bits + 1 unused pad bit
  localparam int unsigned LR_INST_W   = LR_OPCODE_W + LR_FIELD0_W + LR_FIELD1_W + LR_FIELD2_W; // 17
  localparam int unsigned LR_LANES    = 3;   // SLOT_COUNT["lr"] in instruction_spec.py

  // LR-slot opcodes. Position in INSTRUCTION_SPEC["lr"] IS the opcode --
  // never renumber these by hand; if instruction_spec.py's dict order ever
  // changes, this enum must be regenerated to match, not hand-patched.
  typedef enum logic [LR_OPCODE_W-1:0] {
    LR_OP_SET           = 3'd0,
    LR_OP_ADD           = 3'd1,
    LR_OP_SUB           = 3'd2,
    LR_OP_INCR_MOD_POW2 = 3'd3,
    LR_OP_INC           = 3'd4,
    LR_OP_DEC           = 3'd5,
    LR_OP_NOP           = 3'd6
  } lr_opcode_e;

  // ipu_common/incr_mod_pow2_k.py: semantic k in [1,9], encoded as (k-1) in
  // 4 bits (field2's low 4 bits; the 5th bit is the pad bit noted above).
  localparam int unsigned LR_MOD_POW2_K_MIN = 1;
  localparam int unsigned LR_MOD_POW2_K_ENCODED_MAX = 8;  // 9 - 1
  localparam int unsigned LR_MOD_POW2_K_FIELD_W = 4;

  // ipu_common/lr_inc_dec_imm.py: INC/DEC's imm shares field0 with LcrIdx,
  // so its width is field0's width (5 bits, unsigned 0..31) -- not a fixed
  // constant in the Python source (it's resolved from the union layout at
  // import time); recorded here as the concrete number that layout yields
  // today.
  localparam int unsigned LR_INC_DEC_IMM_W = LR_FIELD0_W;

  // -------------------------------------------------------------------
  // COND-slot union layout (ipu_common.union_layout / SLOT_UNIONS["cond"]).
  // Not consumed by this package's LR-only RTL yet (see issue #4) --
  // recorded here now because deriving it already turned up evidence for
  // the IMEM-depth discrepancy tracked in the epic: the Label field is 10
  // bits wide, i.e. it addresses a 1024-entry program space
  // (get_operand_type_bits(): "Label": 10, comment "(MAX_PROGRAM_SIZE - 1)
  // .bit_length() for size 1024") -- matching the *emulator's*
  // INST_MEM_SIZE=1024, not stage-control.md's hardware IMEM_DEPTH=256
  // (which would need only 8 bits) or PC_W=7 (128 entries/bank). The
  // assembler's binary encoding itself is sized for the emulator's flat
  // address space, not the spec's banked hardware one. See
  // planning/ctrl-stage-rtl/01-spec-emulator-reconciliation.md, gap #3.
  // -------------------------------------------------------------------
  localparam int unsigned COND_OPCODE_W = 3;   // 7 opcodes
  localparam int unsigned COND_LABEL_W  = 10;  // field0 -- see note above
  localparam int unsigned COND_LCR_W    = 5;   // field1 (reg1/reg), field2 (reg2)

  typedef enum logic [COND_OPCODE_W-1:0] {
    COND_OP_BEQ  = 3'd0,
    COND_OP_BNE  = 3'd1,
    COND_OP_BLT  = 3'd2,
    COND_OP_BGE  = 3'd3,
    COND_OP_BR   = 3'd4,
    COND_OP_BKPT = 3'd5,
    COND_OP_NOP  = 3'd6
  } cond_opcode_e;

endpackage : ctrl_pkg
