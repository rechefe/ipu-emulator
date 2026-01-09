#ifndef IPU_REGFILE_H
#define IPU_REGFILE_H

#include "ipu_base.h"
#include <stdint.h>

// R register load/store operations
void ipu__load_r_reg(ipu__obj_t *ipu, int rx_idx, int cr_idx, int lr_idx, const ipu__regfile_t *regfile_snapshot);
void ipu__store_r_reg(ipu__obj_t *ipu, int rx_idx, int cr_idx, int lr_idx, const ipu__regfile_t *regfile_snapshot);

// Register clear operations
void ipu__clear_reg(ipu__obj_t *ipu, int rx_idx);
void ipu__clear_rq_reg(ipu__obj_t *ipu, int rq_idx);

// LR register operations
void ipu__set_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm);
void ipu__add_lr(ipu__obj_t *ipu, int lr_idx, uint32_t imm);

// CR register operations
void ipu__set_cr(ipu__obj_t *ipu, int cr_idx, uint32_t imm);

// Helper function for MAC operations
void ipu__get_r_register_for_mac_op(ipu__obj_t *ipu,
                                    int r_reg_index,
                                    const ipu__regfile_t *regfile_snapshot,
                                    ipu__r_reg_t *out_r_reg);

#endif // IPU_REGFILE_H
