#ifndef IPU_REGFILE_H
#define IPU_REGFILE_H

#include "ipu_base.h"
#include "fp/fp.h"
#include <stdint.h>

// R register operations
void ipu__load_r_reg(ipu__obj_t *ipu, int xmem_addr, ipu__r_reg_t *out_r_reg);
void ipu__store_r_reg(ipu__obj_t *ipu, int xmem_addr, ipu__r_reg_t *r_reg);

// LR register operations
void ipu__set_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm);
void ipu__add_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm);

// CR register operations
void ipu__set_cr(ipu__obj_t *ipu, int cr_idx, uint32_t imm);

// R Cyclic operations
void ipu__get_r_cyclic_at_idx(ipu__obj_t *ipu, int start_idx, ipu__r_reg_t *out_r_cyclic_reg);
void ipu__set_r_cyclic_at_idx(ipu__obj_t *ipu, int start_idx, ipu__r_reg_t *in_r_cyclic_reg);

// Accumulator regs operations
void ipu__pack_tf32_array(uint8_t *out_bytes, fp__tf32_t *in_fp32_array, size_t num_elements);
void ipu__unpack_tf32_array(fp__tf32_t *out_fp32_array, const uint8_t *in_bytes, size_t num_elements);

void ipu__set_tf32_reg_in_r_acc(ipu__r_acc_reg_t acc_reg, int r_acc_idx, fp__tf32_t *tf32_value);
void ipu__get_tf32_reg_from_r_acc(ipu__r_acc_reg_t acc_reg, int r_acc_idx, fp__tf32_t *out_tf32_value);

void ipu__get_rt_from_r_acc(ipu__r_acc_reg_t acc_reg, ipu__rt_from_r_acc_t *out_rt_from_r_acc);
void ipu__set_rt_in_r_acc(ipu__r_acc_reg_t acc_reg, ipu__rt_from_r_acc_t *in_rt_from_r_acc);

ipu__r_reg_t *ipu__get_mult_stage_r_reg(ipu__obj_t *ipu, inst_parser__mult_stage_reg_field_t mult_stage_idx);

#endif // IPU_REGFILE_H
