#ifndef IPU_MAC_INST_H
#define IPU_MAC_INST_H

#include "ipu_base.h"
#include "ipu_math/ipu_math.h"
#include "src/tools/ipu-as-py/inst_parser.h"

// MAC instruction execution
void ipu__execute_mac_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot);

// MAC operation functions
void ipu__mac_element_element(ipu__obj_t *ipu,
                              int rx, int ry,
                              ipu_math__dtype_t data_type,
                              const ipu__regfile_t *regfile_snapshot,
                              ipu__rq_reg_t *out_rq_reg);

/**
 * @brief This function makes the IPU do a MAC operation
 * while taking only one byte of the R[ry] reg -
 * RQ[rz][i] += (R[rx][i] * R[ry][LR[lr_idx]])
 * @param r_source_0 the index of the first R register
 * @param r_source_1 the index of the second R register
 * @param lr_idx - selects the LR - according to its value we
 * choose which byte to read from R[ry]
 * @param data_type - which data type to use
 * @param regfile_snapshot - snapshot of register file
 * @param out_rq_reg - output RQ register
 * @return none
 */
void ipu__mac_element_vector(ipu__obj_t *ipu,
                             int r_source_0, int r_source_1,
                             int lr_idx,
                             ipu_math__dtype_t data_type,
                             const ipu__regfile_t *regfile_snapshot,
                             ipu__rq_reg_t *out_rq_reg);

#endif // IPU_MAC_INST_H
