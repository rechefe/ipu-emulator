#ifndef IPU_H
#define IPU_H

#include "ipu_base.h"
#include "ipu_regfile.h"
#include "ipu_xmem_inst.h"
#include "ipu_lr_inst.h"
#include "ipu_mult_inst.h"
#include "ipu_acc_inst.h"
#include "ipu_cond_inst.h"
#include "ipu_break_inst.h"
#include <stdio.h>

// Top-level IPU functions
inst_parser__inst_t ipu__fetch_current_instruction(ipu__obj_t *ipu);

/**
 * @brief Execute the next instruction
 * 
 * @param ipu The IPU object
 * @return IPU_BREAK_RESULT_BREAK if a break instruction triggered, 
 *         IPU_BREAK_RESULT_CONTINUE otherwise
 */
ipu__break_result_t ipu__execute_next_instruction(ipu__obj_t *ipu);

/**
 * @brief Execute the current instruction without checking break
 * 
 * Used after returning from a break prompt to complete the instruction
 * without re-triggering the break.
 * 
 * @param ipu The IPU object
 */
void ipu__execute_instruction_skip_break(ipu__obj_t *ipu);

// Instruction memory management
void ipu__load_inst_mem(ipu__obj_t *ipu, FILE *file);

#endif // IPU_H
