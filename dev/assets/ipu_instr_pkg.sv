// Auto-generated from instruction_spec.py — do not edit.
// Content: sha256:cab08732da174f411a8b5946089ee60bbc192c7d1fa2bb7fbf86d244e5eba30d
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

  typedef enum logic [1:0] {
    AAQ_INST_OPCODE_NOP = 2'd0,
    AAQ_INST_OPCODE_LD_LUT = 2'd1,
    AAQ_INST_OPCODE_ACTIVATE_QUANTIZE = 2'd2
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

  typedef enum logic [4:0] {
    HALF_CR_REG_FIELD_CR0_L = 5'd0,
    HALF_CR_REG_FIELD_CR0_H = 5'd1,
    HALF_CR_REG_FIELD_CR1_L = 5'd2,
    HALF_CR_REG_FIELD_CR1_H = 5'd3,
    HALF_CR_REG_FIELD_CR2_L = 5'd4,
    HALF_CR_REG_FIELD_CR2_H = 5'd5,
    HALF_CR_REG_FIELD_CR3_L = 5'd6,
    HALF_CR_REG_FIELD_CR3_H = 5'd7,
    HALF_CR_REG_FIELD_CR4_L = 5'd8,
    HALF_CR_REG_FIELD_CR4_H = 5'd9,
    HALF_CR_REG_FIELD_CR5_L = 5'd10,
    HALF_CR_REG_FIELD_CR5_H = 5'd11,
    HALF_CR_REG_FIELD_CR6_L = 5'd12,
    HALF_CR_REG_FIELD_CR6_H = 5'd13,
    HALF_CR_REG_FIELD_CR7_L = 5'd14,
    HALF_CR_REG_FIELD_CR7_H = 5'd15,
    HALF_CR_REG_FIELD_CR8_L = 5'd16,
    HALF_CR_REG_FIELD_CR8_H = 5'd17,
    HALF_CR_REG_FIELD_CR9_L = 5'd18,
    HALF_CR_REG_FIELD_CR9_H = 5'd19,
    HALF_CR_REG_FIELD_CR10_L = 5'd20,
    HALF_CR_REG_FIELD_CR10_H = 5'd21,
    HALF_CR_REG_FIELD_CR11_L = 5'd22,
    HALF_CR_REG_FIELD_CR11_H = 5'd23,
    HALF_CR_REG_FIELD_CR12_L = 5'd24,
    HALF_CR_REG_FIELD_CR12_H = 5'd25,
    HALF_CR_REG_FIELD_CR13_L = 5'd26,
    HALF_CR_REG_FIELD_CR13_H = 5'd27,
    HALF_CR_REG_FIELD_CR14_L = 5'd28,
    HALF_CR_REG_FIELD_CR14_H = 5'd29,
    HALF_CR_REG_FIELD_CR15_L = 5'd30,
    HALF_CR_REG_FIELD_CR15_H = 5'd31
  } half_cr_reg_field_t;

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

  typedef enum logic [5:0] {
    LCR_HALF_REG_FIELD_LR0 = 6'd0,
    LCR_HALF_REG_FIELD_LR1 = 6'd1,
    LCR_HALF_REG_FIELD_LR2 = 6'd2,
    LCR_HALF_REG_FIELD_LR3 = 6'd3,
    LCR_HALF_REG_FIELD_LR4 = 6'd4,
    LCR_HALF_REG_FIELD_LR5 = 6'd5,
    LCR_HALF_REG_FIELD_LR6 = 6'd6,
    LCR_HALF_REG_FIELD_LR7 = 6'd7,
    LCR_HALF_REG_FIELD_LR8 = 6'd8,
    LCR_HALF_REG_FIELD_LR9 = 6'd9,
    LCR_HALF_REG_FIELD_LR10 = 6'd10,
    LCR_HALF_REG_FIELD_LR11 = 6'd11,
    LCR_HALF_REG_FIELD_LR12 = 6'd12,
    LCR_HALF_REG_FIELD_LR13 = 6'd13,
    LCR_HALF_REG_FIELD_LR14 = 6'd14,
    LCR_HALF_REG_FIELD_LR15 = 6'd15,
    LCR_HALF_REG_FIELD_CR0 = 6'd16,
    LCR_HALF_REG_FIELD_CR1 = 6'd17,
    LCR_HALF_REG_FIELD_CR2 = 6'd18,
    LCR_HALF_REG_FIELD_CR3 = 6'd19,
    LCR_HALF_REG_FIELD_CR4 = 6'd20,
    LCR_HALF_REG_FIELD_CR5 = 6'd21,
    LCR_HALF_REG_FIELD_CR6 = 6'd22,
    LCR_HALF_REG_FIELD_CR7 = 6'd23,
    LCR_HALF_REG_FIELD_CR8 = 6'd24,
    LCR_HALF_REG_FIELD_CR9 = 6'd25,
    LCR_HALF_REG_FIELD_CR10 = 6'd26,
    LCR_HALF_REG_FIELD_CR11 = 6'd27,
    LCR_HALF_REG_FIELD_CR12 = 6'd28,
    LCR_HALF_REG_FIELD_CR13 = 6'd29,
    LCR_HALF_REG_FIELD_CR14 = 6'd30,
    LCR_HALF_REG_FIELD_CR15 = 6'd31,
    LCR_HALF_REG_FIELD_CR0_L = 6'd32,
    LCR_HALF_REG_FIELD_CR1_L = 6'd33,
    LCR_HALF_REG_FIELD_CR2_L = 6'd34,
    LCR_HALF_REG_FIELD_CR3_L = 6'd35,
    LCR_HALF_REG_FIELD_CR4_L = 6'd36,
    LCR_HALF_REG_FIELD_CR5_L = 6'd37,
    LCR_HALF_REG_FIELD_CR6_L = 6'd38,
    LCR_HALF_REG_FIELD_CR7_L = 6'd39,
    LCR_HALF_REG_FIELD_CR8_L = 6'd40,
    LCR_HALF_REG_FIELD_CR9_L = 6'd41,
    LCR_HALF_REG_FIELD_CR10_L = 6'd42,
    LCR_HALF_REG_FIELD_CR11_L = 6'd43,
    LCR_HALF_REG_FIELD_CR12_L = 6'd44,
    LCR_HALF_REG_FIELD_CR13_L = 6'd45,
    LCR_HALF_REG_FIELD_CR14_L = 6'd46,
    LCR_HALF_REG_FIELD_CR15_L = 6'd47,
    LCR_HALF_REG_FIELD_CR0_H = 6'd48,
    LCR_HALF_REG_FIELD_CR1_H = 6'd49,
    LCR_HALF_REG_FIELD_CR2_H = 6'd50,
    LCR_HALF_REG_FIELD_CR3_H = 6'd51,
    LCR_HALF_REG_FIELD_CR4_H = 6'd52,
    LCR_HALF_REG_FIELD_CR5_H = 6'd53,
    LCR_HALF_REG_FIELD_CR6_H = 6'd54,
    LCR_HALF_REG_FIELD_CR7_H = 6'd55,
    LCR_HALF_REG_FIELD_CR8_H = 6'd56,
    LCR_HALF_REG_FIELD_CR9_H = 6'd57,
    LCR_HALF_REG_FIELD_CR10_H = 6'd58,
    LCR_HALF_REG_FIELD_CR11_H = 6'd59,
    LCR_HALF_REG_FIELD_CR12_H = 6'd60,
    LCR_HALF_REG_FIELD_CR13_H = 6'd61,
    LCR_HALF_REG_FIELD_CR14_H = 6'd62,
    LCR_HALF_REG_FIELD_CR15_H = 6'd63
  } lcr_half_reg_field_t;

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

  typedef enum logic [2:0] {
    AAQ_ACTIVATION_FN_FIELD_IDENTITY = 3'd0,
    AAQ_ACTIVATION_FN_FIELD_RELU = 3'd1,
    AAQ_ACTIVATION_FN_FIELD_RELU6 = 3'd2,
    AAQ_ACTIVATION_FN_FIELD_RECIPROCAL = 3'd3,
    AAQ_ACTIVATION_FN_FIELD_RSQRT = 3'd4,
    AAQ_ACTIVATION_FN_FIELD_EXP2 = 3'd5,
    AAQ_ACTIVATION_FN_FIELD_GENERIC = 3'd6
  } aaq_activation_fn_field_t;

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

  localparam int unsigned LR_SLOT_U_WIDTH = 16;

  // Operand payload only (opcode is in lr_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [1:0] padding_0; // padding
      lcr_half_reg_field_t src_0; // LcrHalfIdx
      lr_reg_field_t reg_1; // LrIdx
      logic [3:0] padding_2; // padding
    } set_inst;
    struct packed {
      logic [1:0] padding_0; // padding
      lcr_half_reg_field_t src_b_0; // LcrHalfIdx
      lr_reg_field_t dest_1; // LrIdx
      lr_reg_field_t src_a_2; // LrIdx
    } add;
    struct packed {
      logic [1:0] padding_0; // padding
      lcr_half_reg_field_t src_b_0; // LcrHalfIdx
      lr_reg_field_t dest_1; // LrIdx
      lr_reg_field_t src_a_2; // LrIdx
    } sub;
    struct packed {
      logic [1:0] padding_0; // padding
      lcr_half_reg_field_t step_0; // LcrHalfIdx
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
      logic [1:0] padding_0; // padding
      lcr_half_reg_field_t src_b_0; // LcrHalfIdx
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

  localparam int unsigned MULT_SLOT_U_WIDTH = 26;

  // Operand payload only (opcode is in mult_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [3:0] padding_0; // padding
      mult_stage_reg_field_t ra_0; // MultStageReg
      half_cr_reg_field_t ds_1; // HalfCrIdx
      half_cr_reg_field_t qf_2; // HalfCrIdx
      lr_reg_field_t rc_idx_3; // LrIdx
      lr_reg_field_t mask_shift_4; // LrIdx
      logic [2:0] mask_offset_5; // MultMaskOffsetImmediate
    } mult_rc_vv;
    struct packed {
      lcr_reg_field_t src_0; // LcrIdx
      half_cr_reg_field_t ds_1; // HalfCrIdx
      half_cr_reg_field_t qf_2; // HalfCrIdx
      lr_reg_field_t rc_idx_3; // LrIdx
      lr_reg_field_t mask_shift_4; // LrIdx
      logic [2:0] mask_offset_5; // MultMaskOffsetImmediate
    } mult_rc_ve;
    struct packed {
      logic [4:0] padding_0; // padding
      half_cr_reg_field_t ds_1; // HalfCrIdx
      half_cr_reg_field_t qf_2; // HalfCrIdx
      lr_reg_field_t rc_idx_3; // LrIdx
      lr_reg_field_t mask_shift_4; // LrIdx
      logic [2:0] mask_offset_5; // MultMaskOffsetImmediate
    } mult_rc_vs;
    struct packed {
      logic [25:0] padding; // padding
    } nop;
    struct packed {
      logic [0:0] padding_0; // padding
      cr_reg_field_t cr_idx_0; // CrIdx
      half_cr_reg_field_t ds_1; // HalfCrIdx
      half_cr_reg_field_t qf_2; // HalfCrIdx
      lr_reg_field_t ra_idx_3; // LrIdx
      lr_reg_field_t mask_shift_4; // LrIdx
      logic [2:0] mask_offset_5; // MultMaskOffsetImmediate
    } mult_ve;
    struct packed {
      logic [0:0] padding_0; // padding
      cr_reg_field_t cr_idx_0; // CrIdx
      half_cr_reg_field_t ds_1; // HalfCrIdx
      half_cr_reg_field_t qf_2; // HalfCrIdx
      lr_reg_field_t ra_idx_3; // LrIdx
      lr_reg_field_t mask_shift_4; // LrIdx
      logic [2:0] mask_offset_5; // MultMaskOffsetImmediate
    } mult_ee;
  } mult_slot_u;

  typedef struct packed {
    mult_inst_opcode_t opcode;
    mult_slot_u operands;
  } mult_slot_t;

  localparam int unsigned MULT_SLOT_T_WIDTH = 29;

  localparam int unsigned ACC_SLOT_U_WIDTH = 14;

  // Operand payload only (opcode is in acc_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [13:0] padding; // padding
    } acc_add;
    struct packed {
      logic [13:0] padding; // padding
    } acc_add_first;
    struct packed {
      logic [13:0] padding; // padding
    } acc_max;
    struct packed {
      logic [13:0] padding; // padding
    } acc_max_first;
    struct packed {
      logic [13:0] padding; // padding
    } acc_sub;
    struct packed {
      logic [13:0] padding; // padding
    } acc_sub_first;
    struct packed {
      logic [13:0] padding; // padding
    } nop;
    struct packed {
      logic [2:0] padding_0; // padding
      elements_in_row_field_t elements_in_row_0; // ElementsInRow
      logic [0:0] padding_1; // padding
      horizontal_stride_field_t horizontal_stride_1; // HorizontalStride
      lr_reg_field_t offset_2; // LrIdx
      vertical_stride_field_t vertical_stride_3; // VerticalStride
    } acc_stride;
    struct packed {
      half_cr_reg_field_t ds_0; // HalfCrIdx
      logic [2:0] padding_1; // padding
      lr_reg_field_t dest_slot_2; // LrIdx
      logic [1:0] padding_3; // padding
    } agg_sum_first;
    struct packed {
      half_cr_reg_field_t ds_0; // HalfCrIdx
      logic [2:0] padding_1; // padding
      lr_reg_field_t dest_slot_2; // LrIdx
      logic [1:0] padding_3; // padding
    } agg_sum;
    struct packed {
      half_cr_reg_field_t ds_0; // HalfCrIdx
      logic [2:0] padding_1; // padding
      lr_reg_field_t dest_slot_2; // LrIdx
      logic [1:0] padding_3; // padding
    } agg_max_first;
    struct packed {
      half_cr_reg_field_t ds_0; // HalfCrIdx
      logic [2:0] padding_1; // padding
      lr_reg_field_t dest_slot_2; // LrIdx
      logic [1:0] padding_3; // padding
    } agg_max;
    struct packed {
      logic [4:0] reshape_mask_0; // LrOrReshapeMaskImmediate
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

  localparam int unsigned ACC_SLOT_T_WIDTH = 18;

  localparam int unsigned AAQ_SLOT_U_WIDTH = 21;

  // Operand payload only (opcode is in aaq_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [20:0] padding; // padding
    } nop;
    struct packed {
      logic [2:0] lut_addr_0; // LutAddrImmediate
      cr_reg_field_t table_id_1; // CrIdx
      logic [4:0] padding_2; // padding
      logic [4:0] padding_3; // padding
      lr_reg_field_t table_offset_4; // LrIdx
    } ld_lut;
    struct packed {
      aaq_activation_fn_field_t activation_fn_0; // AaqActivationFn
      cr_reg_field_t table_id_1; // CrIdx
      half_cr_reg_field_t ds_2; // HalfCrIdx
      half_cr_reg_field_t qf_3; // HalfCrIdx
      lr_reg_field_t table_offset_4; // LrIdx
    } activate_quantize;
  } aaq_slot_u;

  typedef struct packed {
    aaq_inst_opcode_t opcode;
    aaq_slot_u operands;
  } aaq_slot_t;

  localparam int unsigned AAQ_SLOT_T_WIDTH = 23;

  localparam int unsigned COND_SLOT_U_WIDTH = 22;

  // Operand payload only (opcode is in cond_slot_t.opcode, shared across members)
  typedef union packed {
    struct packed {
      logic [9:0] label_0; // Label
      lcr_half_reg_field_t reg1_1; // LcrHalfIdx
      lcr_half_reg_field_t reg2_2; // LcrHalfIdx
    } beq;
    struct packed {
      logic [9:0] label_0; // Label
      lcr_half_reg_field_t reg1_1; // LcrHalfIdx
      lcr_half_reg_field_t reg2_2; // LcrHalfIdx
    } bne;
    struct packed {
      logic [9:0] label_0; // Label
      lcr_half_reg_field_t reg1_1; // LcrHalfIdx
      lcr_half_reg_field_t reg2_2; // LcrHalfIdx
    } blt_s;
    struct packed {
      logic [9:0] label_0; // Label
      lcr_half_reg_field_t reg1_1; // LcrHalfIdx
      lcr_half_reg_field_t reg2_2; // LcrHalfIdx
    } bge_s;
    struct packed {
      logic [9:0] padding_0; // padding
      lcr_half_reg_field_t reg_1; // LcrHalfIdx
      logic [5:0] padding_2; // padding
    } br;
    struct packed {
      logic [21:0] padding; // padding
    } halt;
    struct packed {
      logic [21:0] padding; // padding
    } nop;
  } cond_slot_u;

  typedef struct packed {
    cond_inst_opcode_t opcode;
    cond_slot_u operands;
  } cond_slot_t;

  localparam int unsigned COND_SLOT_T_WIDTH = 25;

  // -------------------------------------------------------------------------
  // Compound VLIW instruction (slots MSB → LSB: cond, lr×3, load, mult, acc, aaq)
  // This is the hardware word. The emulator word appends the emulator-only
  // slots (aaq_exact, break) below bit 0 of it; they are not in this package.
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
  } ipu_compound_inst_t;

  localparam int unsigned IPU_COMPOUND_INST_WIDTH = 169;

endpackage : ipu_instr_pkg
