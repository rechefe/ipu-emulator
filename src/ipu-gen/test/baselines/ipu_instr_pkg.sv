// Auto-generated from instruction_spec.py — do not edit.
// Content: sha256:25ee36b9e6fb68392a907f9ad614db2f6e00f452a267449f8ecee027dac6f970
//
// Opcode/operand enums and per-slot structs/unions are generated from
// instruction_spec.py (union field layout in SLOT_UNIONS).
package ipu_instr_pkg;

  // -------------------------------------------------------------------------
  // Enumerations (EnumToken descriptors — opcodes, registers, immediates)
  // -------------------------------------------------------------------------
  typedef enum logic [1:0] {
    LOAD_INST_OPCODE_LD_MULT = 2'd0,
    LOAD_INST_OPCODE_LD_MULT_CYCLIC = 2'd1,
    LOAD_INST_OPCODE_LD_MASK = 2'd2,
    LOAD_INST_OPCODE_NOP = 2'd3
  } load_inst_opcode_t;

  typedef enum logic [0:0] {
    STORE_INST_OPCODE_ST_AAQ = 1'd0,
    STORE_INST_OPCODE_NOP = 1'd1
  } store_inst_opcode_t;

  typedef enum logic [3:0] {
    LR_INST_OPCODE_SET = 4'd0,
    LR_INST_OPCODE_ADD = 4'd1,
    LR_INST_OPCODE_SUB = 4'd2,
    LR_INST_OPCODE_INC_MODP2 = 4'd3,
    LR_INST_OPCODE_INC = 4'd4,
    LR_INST_OPCODE_DEC = 4'd5,
    LR_INST_OPCODE_ADDB = 4'd6,
    LR_INST_OPCODE_ADDBI = 4'd7,
    LR_INST_OPCODE_NOP = 4'd8
  } lr_inst_opcode_t;

  typedef enum logic [2:0] {
    MULT_INST_OPCODE_MULT_RC_VV = 3'd0,
    MULT_INST_OPCODE_MULT_RC_VE = 3'd1,
    MULT_INST_OPCODE_MULT_RC_VS = 3'd2,
    MULT_INST_OPCODE_NOP = 3'd3,
    MULT_INST_OPCODE_MULT_VE = 3'd4,
    MULT_INST_OPCODE_MULT_EE = 3'd5
  } mult_inst_opcode_t;

  typedef enum logic [3:0] {
    ACC_INST_OPCODE_ACC_ADD = 4'd0,
    ACC_INST_OPCODE_ACC_ADD_FIRST = 4'd1,
    ACC_INST_OPCODE_ACC_MAX = 4'd2,
    ACC_INST_OPCODE_ACC_MAX_FIRST = 4'd3,
    ACC_INST_OPCODE_ACC_SUB = 4'd4,
    ACC_INST_OPCODE_ACC_SUB_FIRST = 4'd5,
    ACC_INST_OPCODE_NOP = 4'd6,
    ACC_INST_OPCODE_ACC_STRIDE = 4'd7,
    ACC_INST_OPCODE_AGG_SUM_FIRST = 4'd8,
    ACC_INST_OPCODE_AGG_SUM = 4'd9,
    ACC_INST_OPCODE_AGG_MAX_FIRST = 4'd10,
    ACC_INST_OPCODE_AGG_MAX = 4'd11,
    ACC_INST_OPCODE_ACC_RESHAPE = 4'd12
  } acc_inst_opcode_t;

  typedef enum logic [0:0] {
    AAQ_INST_OPCODE_NOP = 1'd0,
    AAQ_INST_OPCODE_ACTIVATE_QUANTIZE = 1'd1
  } aaq_inst_opcode_t;

  typedef enum logic [2:0] {
    COND_INST_OPCODE_BEQ = 3'd0,
    COND_INST_OPCODE_BNE = 3'd1,
    COND_INST_OPCODE_BLT_S = 3'd2,
    COND_INST_OPCODE_BGE_S = 3'd3,
    COND_INST_OPCODE_BR = 3'd4,
    COND_INST_OPCODE_HALT = 3'd5,
    COND_INST_OPCODE_NOP = 3'd6
  } cond_inst_opcode_t;

  typedef enum logic [0:0] {
    MULT_STAGE_REG_FIELD_R0 = 1'd0,
    MULT_STAGE_REG_FIELD_R1 = 1'd1
  } mult_stage_reg_field_t;

  typedef enum logic [3:0] {
    LR_REG_FIELD_LR0 = 4'd0,
    LR_REG_FIELD_LR1 = 4'd1,
    LR_REG_FIELD_LR2 = 4'd2,
    LR_REG_FIELD_LR3 = 4'd3,
    LR_REG_FIELD_LR4 = 4'd4,
    LR_REG_FIELD_LR5 = 4'd5,
    LR_REG_FIELD_LR6 = 4'd6,
    LR_REG_FIELD_LR7 = 4'd7,
    LR_REG_FIELD_LR8 = 4'd8,
    LR_REG_FIELD_LR9 = 4'd9,
    LR_REG_FIELD_LR10 = 4'd10,
    LR_REG_FIELD_LR11 = 4'd11,
    LR_REG_FIELD_LR12 = 4'd12,
    LR_REG_FIELD_LR13 = 4'd13,
    LR_REG_FIELD_LR14 = 4'd14,
    LR_REG_FIELD_LR15 = 4'd15
  } lr_reg_field_t;

  typedef enum logic [3:0] {
    CR_REG_FIELD_CR0 = 4'd0,
    CR_REG_FIELD_CR1 = 4'd1,
    CR_REG_FIELD_CR2 = 4'd2,
    CR_REG_FIELD_CR3 = 4'd3,
    CR_REG_FIELD_CR4 = 4'd4,
    CR_REG_FIELD_CR5 = 4'd5,
    CR_REG_FIELD_CR6 = 4'd6,
    CR_REG_FIELD_CR7 = 4'd7,
    CR_REG_FIELD_CR8 = 4'd8,
    CR_REG_FIELD_CR9 = 4'd9,
    CR_REG_FIELD_CR10 = 4'd10,
    CR_REG_FIELD_CR11 = 4'd11,
    CR_REG_FIELD_CR12 = 4'd12,
    CR_REG_FIELD_CR13 = 4'd13,
    CR_REG_FIELD_CR14 = 4'd14,
    CR_REG_FIELD_CR15 = 4'd15
  } cr_reg_field_t;

  typedef enum logic [3:0] {
    DSTRUCTURE_CR_REG_FIELD_CR0 = 4'd0,
    DSTRUCTURE_CR_REG_FIELD_CR1 = 4'd1,
    DSTRUCTURE_CR_REG_FIELD_CR2 = 4'd2,
    DSTRUCTURE_CR_REG_FIELD_CR3 = 4'd3,
    DSTRUCTURE_CR_REG_FIELD_CR4 = 4'd4,
    DSTRUCTURE_CR_REG_FIELD_CR5 = 4'd5,
    DSTRUCTURE_CR_REG_FIELD_CR6 = 4'd6,
    DSTRUCTURE_CR_REG_FIELD_CR7 = 4'd7,
    DSTRUCTURE_CR_REG_FIELD_CR8 = 4'd8,
    DSTRUCTURE_CR_REG_FIELD_CR9 = 4'd9,
    DSTRUCTURE_CR_REG_FIELD_CR10 = 4'd10,
    DSTRUCTURE_CR_REG_FIELD_CR11 = 4'd11,
    DSTRUCTURE_CR_REG_FIELD_CR12 = 4'd12,
    DSTRUCTURE_CR_REG_FIELD_CR13 = 4'd13,
    DSTRUCTURE_CR_REG_FIELD_CR14 = 4'd14,
    DSTRUCTURE_CR_REG_FIELD_CR15 = 4'd15
  } dstructure_cr_reg_field_t;

  typedef enum logic [4:0] {
    LCR_REG_FIELD_LR0 = 5'd0,
    LCR_REG_FIELD_LR1 = 5'd1,
    LCR_REG_FIELD_LR2 = 5'd2,
    LCR_REG_FIELD_LR3 = 5'd3,
    LCR_REG_FIELD_LR4 = 5'd4,
    LCR_REG_FIELD_LR5 = 5'd5,
    LCR_REG_FIELD_LR6 = 5'd6,
    LCR_REG_FIELD_LR7 = 5'd7,
    LCR_REG_FIELD_LR8 = 5'd8,
    LCR_REG_FIELD_LR9 = 5'd9,
    LCR_REG_FIELD_LR10 = 5'd10,
    LCR_REG_FIELD_LR11 = 5'd11,
    LCR_REG_FIELD_LR12 = 5'd12,
    LCR_REG_FIELD_LR13 = 5'd13,
    LCR_REG_FIELD_LR14 = 5'd14,
    LCR_REG_FIELD_LR15 = 5'd15,
    LCR_REG_FIELD_CR0 = 5'd16,
    LCR_REG_FIELD_CR1 = 5'd17,
    LCR_REG_FIELD_CR2 = 5'd18,
    LCR_REG_FIELD_CR3 = 5'd19,
    LCR_REG_FIELD_CR4 = 5'd20,
    LCR_REG_FIELD_CR5 = 5'd21,
    LCR_REG_FIELD_CR6 = 5'd22,
    LCR_REG_FIELD_CR7 = 5'd23,
    LCR_REG_FIELD_CR8 = 5'd24,
    LCR_REG_FIELD_CR9 = 5'd25,
    LCR_REG_FIELD_CR10 = 5'd26,
    LCR_REG_FIELD_CR11 = 5'd27,
    LCR_REG_FIELD_CR12 = 5'd28,
    LCR_REG_FIELD_CR13 = 5'd29,
    LCR_REG_FIELD_CR14 = 5'd30,
    LCR_REG_FIELD_CR15 = 5'd31
  } lcr_reg_field_t;

  typedef enum logic [2:0] {
    LRD_REG_FIELD_LRD0 = 3'd0,
    LRD_REG_FIELD_LRD2 = 3'd1,
    LRD_REG_FIELD_LRD4 = 3'd2,
    LRD_REG_FIELD_LRD6 = 3'd3,
    LRD_REG_FIELD_LRD8 = 3'd4,
    LRD_REG_FIELD_LRD10 = 3'd5,
    LRD_REG_FIELD_LRD12 = 3'd6,
    LRD_REG_FIELD_LRD14 = 3'd7
  } lrd_reg_field_t;

  typedef enum logic [3:0] {
    ACTIVATION_FN_FIELD_IDENTITY = 4'd0,
    ACTIVATION_FN_FIELD_RELU = 4'd1,
    ACTIVATION_FN_FIELD_RELU6 = 4'd2,
    ACTIVATION_FN_FIELD_SIGMOID = 4'd3,
    ACTIVATION_FN_FIELD_TANH = 4'd4,
    ACTIVATION_FN_FIELD_GELU = 4'd5,
    ACTIVATION_FN_FIELD_SOFTPLUS = 4'd6,
    ACTIVATION_FN_FIELD_ELU = 4'd7,
    ACTIVATION_FN_FIELD_EXP2 = 4'd8,
    ACTIVATION_FN_FIELD_RECIPROCAL = 4'd9,
    ACTIVATION_FN_FIELD_RSQRT = 4'd10,
    ACTIVATION_FN_FIELD_SILU = 4'd11,
    ACTIVATION_FN_FIELD_WINDOW = 4'd12
  } activation_fn_field_t;

  typedef enum logic [1:0] {
    ELEMENTS_IN_ROW_FIELD_16 = 2'd0,
    ELEMENTS_IN_ROW_FIELD_32 = 2'd1,
    ELEMENTS_IN_ROW_FIELD_64 = 2'd2
  } elements_in_row_field_t;

  typedef enum logic [1:0] {
    HORIZONTAL_STRIDE_FIELD_OFF = 2'd0,
    HORIZONTAL_STRIDE_FIELD_ON = 2'd1,
    HORIZONTAL_STRIDE_FIELD_ON_INV = 2'd2,
    HORIZONTAL_STRIDE_FIELD_RESERVED3 = 2'd3
  } horizontal_stride_field_t;

  typedef enum logic [1:0] {
    VERTICAL_STRIDE_FIELD_OFF = 2'd0,
    VERTICAL_STRIDE_FIELD_ON = 2'd1,
    VERTICAL_STRIDE_FIELD_ON_INV = 2'd2,
    VERTICAL_STRIDE_FIELD_RESERVED3 = 2'd3
  } vertical_stride_field_t;

  // -------------------------------------------------------------------------
  // Per-slot encoded words (opcode + operand union, MSB-first within slot)
  // Opcode type for each slot is one of the enums above (e.g. load_inst_opcode_t).
  // -------------------------------------------------------------------------
  localparam int unsigned LOAD_SLOT_U_WIDTH = 12;

  // Operand payload only (opcode is in load_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      cr_reg_field_t base_0; // CrIdx
      lr_reg_field_t offset_1; // LrIdx
      logic [2:0] padding_2; // padding
      mult_stage_reg_field_t dest_2; // MultStageReg
    } ld_mult;
    struct packed {
      cr_reg_field_t base_0; // CrIdx
      lr_reg_field_t offset_1; // LrIdx
      lr_reg_field_t index_2; // LrIdx
    } ld_mult_cyclic;
    struct packed {
      cr_reg_field_t base_0; // CrIdx
      lr_reg_field_t offset_1; // LrIdx
      logic [3:0] padding_2; // padding
    } ld_mask;
    struct packed {
      logic [11:0] padding; // padding
    } nop;
  } load_slot_u;

  typedef struct packed {
    load_inst_opcode_t opcode;
    load_slot_u operands;
  } load_slot_t;

  localparam int unsigned LOAD_SLOT_T_WIDTH = 14;

  localparam int unsigned STORE_SLOT_U_WIDTH = 8;

  // Operand payload only (opcode is in store_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      cr_reg_field_t base_0; // CrIdx
      lr_reg_field_t offset_1; // LrIdx
    } st_aaq;
    struct packed {
      logic [7:0] padding; // padding
    } nop;
  } store_slot_u;

  typedef struct packed {
    store_inst_opcode_t opcode;
    store_slot_u operands;
  } store_slot_t;

  localparam int unsigned STORE_SLOT_T_WIDTH = 9;

  localparam int unsigned LR_SLOT_U_WIDTH = 16;

  // Operand payload only (opcode is in lr_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [3:0] padding_0; // padding
      cr_reg_field_t src_0; // CrIdx
      lr_reg_field_t reg_1; // LrIdx
      logic [3:0] padding_2; // padding
    } set_inst;
    struct packed {
      logic [2:0] padding_0; // padding
      lcr_reg_field_t src_b_0; // LcrIdx
      lr_reg_field_t dest_1; // LrIdx
      lr_reg_field_t src_a_2; // LrIdx
    } add;
    struct packed {
      logic [2:0] padding_0; // padding
      lcr_reg_field_t src_b_0; // LcrIdx
      lr_reg_field_t dest_1; // LrIdx
      lr_reg_field_t src_a_2; // LrIdx
    } sub;
    struct packed {
      logic [2:0] padding_0; // padding
      lcr_reg_field_t step_0; // LcrIdx
      lr_reg_field_t dest_1; // LrIdx
      logic [3:0] k_2; // LrModPow2KImmediate
    } inc_modp2;
    struct packed {
      logic [7:0] imm_0; // LrIncDecImmediate
      lr_reg_field_t dest_1; // LrIdx
      logic [3:0] padding_2; // padding
    } inc;
    struct packed {
      logic [7:0] imm_0; // LrIncDecImmediate
      lr_reg_field_t dest_1; // LrIdx
      logic [3:0] padding_2; // padding
    } dec;
    struct packed {
      logic [2:0] padding_0; // padding
      lcr_reg_field_t src_b_0; // LcrIdx
      logic [0:0] padding_1; // padding
      lrd_reg_field_t dest_1; // LrdIdx
      logic [3:0] padding_2; // padding
    } addb;
    struct packed {
      logic [7:0] imm_0; // AddbiImmediate
      logic [0:0] padding_1; // padding
      lrd_reg_field_t dest_1; // LrdIdx
      logic [3:0] padding_2; // padding
    } addbi;
    struct packed {
      logic [15:0] padding; // padding
    } nop;
  } lr_slot_u;

  typedef struct packed {
    lr_inst_opcode_t opcode;
    lr_slot_u operands;
  } lr_slot_t;

  localparam int unsigned LR_SLOT_T_WIDTH = 20;

  localparam int unsigned MULT_SLOT_U_WIDTH = 20;

  // Operand payload only (opcode is in mult_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [3:0] padding_0; // padding
      mult_stage_reg_field_t ra_0; // MultStageReg
      dstructure_cr_reg_field_t cr_idx_1; // DstructureCrIdx
      lr_reg_field_t rc_idx_2; // LrIdx
      lr_reg_field_t mask_shift_3; // LrIdx
      logic [2:0] mask_offset_4; // MultMaskOffsetImmediate
    } mult_rc_vv;
    struct packed {
      lcr_reg_field_t src_0; // LcrIdx
      dstructure_cr_reg_field_t cr_idx_1; // DstructureCrIdx
      lr_reg_field_t rc_idx_2; // LrIdx
      lr_reg_field_t mask_shift_3; // LrIdx
      logic [2:0] mask_offset_4; // MultMaskOffsetImmediate
    } mult_rc_ve;
    struct packed {
      logic [4:0] padding_0; // padding
      dstructure_cr_reg_field_t cr_idx_1; // DstructureCrIdx
      lr_reg_field_t rc_idx_2; // LrIdx
      lr_reg_field_t mask_shift_3; // LrIdx
      logic [2:0] mask_offset_4; // MultMaskOffsetImmediate
    } mult_rc_vs;
    struct packed {
      logic [19:0] padding; // padding
    } nop;
    struct packed {
      logic [0:0] padding_0; // padding
      cr_reg_field_t cr_idx_0; // CrIdx
      dstructure_cr_reg_field_t dstructure_cr_idx_1; // DstructureCrIdx
      lr_reg_field_t ra_idx_2; // LrIdx
      lr_reg_field_t mask_shift_3; // LrIdx
      logic [2:0] mask_offset_4; // MultMaskOffsetImmediate
    } mult_ve;
    struct packed {
      logic [0:0] padding_0; // padding
      cr_reg_field_t cr_idx_0; // CrIdx
      dstructure_cr_reg_field_t dstructure_cr_idx_1; // DstructureCrIdx
      lr_reg_field_t ra_idx_2; // LrIdx
      lr_reg_field_t mask_shift_3; // LrIdx
      logic [2:0] mask_offset_4; // MultMaskOffsetImmediate
    } mult_ee;
  } mult_slot_u;

  typedef struct packed {
    mult_inst_opcode_t opcode;
    mult_slot_u operands;
  } mult_slot_t;

  localparam int unsigned MULT_SLOT_T_WIDTH = 23;

  localparam int unsigned ACC_SLOT_U_WIDTH = 13;

  // Operand payload only (opcode is in acc_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [12:0] padding; // padding
    } acc_add;
    struct packed {
      logic [12:0] padding; // padding
    } acc_add_first;
    struct packed {
      logic [12:0] padding; // padding
    } acc_max;
    struct packed {
      logic [12:0] padding; // padding
    } acc_max_first;
    struct packed {
      logic [12:0] padding; // padding
    } acc_sub;
    struct packed {
      logic [12:0] padding; // padding
    } acc_sub_first;
    struct packed {
      logic [12:0] padding; // padding
    } nop;
    struct packed {
      logic [1:0] padding_0; // padding
      elements_in_row_field_t elements_in_row_0; // ElementsInRow
      logic [0:0] padding_1; // padding
      horizontal_stride_field_t horizontal_stride_1; // HorizontalStride
      lr_reg_field_t offset_2; // LrIdx
      vertical_stride_field_t vertical_stride_3; // VerticalStride
    } acc_stride;
    struct packed {
      dstructure_cr_reg_field_t cr_idx_0; // DstructureCrIdx
      logic [2:0] padding_1; // padding
      lr_reg_field_t dest_slot_2; // LrIdx
      logic [1:0] padding_3; // padding
    } agg_sum_first;
    struct packed {
      dstructure_cr_reg_field_t cr_idx_0; // DstructureCrIdx
      logic [2:0] padding_1; // padding
      lr_reg_field_t dest_slot_2; // LrIdx
      logic [1:0] padding_3; // padding
    } agg_sum;
    struct packed {
      dstructure_cr_reg_field_t cr_idx_0; // DstructureCrIdx
      logic [2:0] padding_1; // padding
      lr_reg_field_t dest_slot_2; // LrIdx
      logic [1:0] padding_3; // padding
    } agg_max_first;
    struct packed {
      dstructure_cr_reg_field_t cr_idx_0; // DstructureCrIdx
      logic [2:0] padding_1; // padding
      lr_reg_field_t dest_slot_2; // LrIdx
      logic [1:0] padding_3; // padding
    } agg_max;
    struct packed {
      logic [3:0] reshape_mask_0; // LrOrReshapeMaskImmediate
      lrd_reg_field_t source_1; // LrdIdx
      logic [0:0] padding_2; // padding
      lrd_reg_field_t dest_2; // LrdIdx
      logic [1:0] padding_3; // padding
    } acc_reshape;
  } acc_slot_u;

  typedef struct packed {
    acc_inst_opcode_t opcode;
    acc_slot_u operands;
  } acc_slot_t;

  localparam int unsigned ACC_SLOT_T_WIDTH = 17;

  localparam int unsigned AAQ_SLOT_U_WIDTH = 8;

  // Operand payload only (opcode is in aaq_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [7:0] padding; // padding
    } nop;
    struct packed {
      activation_fn_field_t activation_fn_0; // ActivationFn
      dstructure_cr_reg_field_t cr_idx_1; // DstructureCrIdx
    } activate_quantize;
  } aaq_slot_u;

  typedef struct packed {
    aaq_inst_opcode_t opcode;
    aaq_slot_u operands;
  } aaq_slot_t;

  localparam int unsigned AAQ_SLOT_T_WIDTH = 9;

  localparam int unsigned COND_SLOT_U_WIDTH = 20;

  // Operand payload only (opcode is in cond_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [9:0] label_0; // Label
      lcr_reg_field_t reg1_1; // LcrIdx
      lcr_reg_field_t reg2_2; // LcrIdx
    } beq;
    struct packed {
      logic [9:0] label_0; // Label
      lcr_reg_field_t reg1_1; // LcrIdx
      lcr_reg_field_t reg2_2; // LcrIdx
    } bne;
    struct packed {
      logic [9:0] label_0; // Label
      lcr_reg_field_t reg1_1; // LcrIdx
      lcr_reg_field_t reg2_2; // LcrIdx
    } blt_s;
    struct packed {
      logic [9:0] label_0; // Label
      lcr_reg_field_t reg1_1; // LcrIdx
      lcr_reg_field_t reg2_2; // LcrIdx
    } bge_s;
    struct packed {
      logic [9:0] padding_0; // padding
      lcr_reg_field_t reg_1; // LcrIdx
      logic [4:0] padding_2; // padding
    } br;
    struct packed {
      logic [19:0] padding; // padding
    } halt;
    struct packed {
      logic [19:0] padding; // padding
    } nop;
  } cond_slot_u;

  typedef struct packed {
    cond_inst_opcode_t opcode;
    cond_slot_u operands;
  } cond_slot_t;

  localparam int unsigned COND_SLOT_T_WIDTH = 23;

  // -------------------------------------------------------------------------
  // Compound VLIW instruction (slots MSB → LSB: cond, lr×3, load, mult, acc, aaq, store)
  // This is the hardware word. The emulator word appends the emulator-only
  // slots (acc_store, break) below bit 0 of it; they are not in this package.
  // -------------------------------------------------------------------------
  typedef struct packed {
    cond_slot_t cond_slot;
    lr_slot_t lr_slot_0;
    lr_slot_t lr_slot_1;
    lr_slot_t lr_slot_2;
    load_slot_t load_slot;
    mult_slot_t mult_slot;
    acc_slot_t acc_slot;
    aaq_slot_t aaq_slot;
    store_slot_t store_slot;
  } ipu_compound_inst_t;

  localparam int unsigned IPU_COMPOUND_INST_WIDTH = 155;

endpackage : ipu_instr_pkg
