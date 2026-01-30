#ifndef IPU_BREAK_INST_H
#define IPU_BREAK_INST_H

#include "ipu_base.h"
#include "src/tools/ipu-as-py/inst_parser.h"
#include <stdbool.h>

/**
 * @brief Result of executing a break instruction
 */
typedef enum {
    IPU_BREAK_RESULT_CONTINUE,  // No break triggered, continue execution
    IPU_BREAK_RESULT_BREAK,     // Break triggered, should halt and enter debug
} ipu__break_result_t;

/**
 * @brief Execute break instruction and check if execution should halt
 * 
 * This function should be called BEFORE any other instruction handlers
 * to ensure break triggers before any side effects.
 * 
 * @param ipu The IPU object
 * @param inst The current instruction
 * @param regfile_snapshot Snapshot of register file at cycle start
 * @return IPU_BREAK_RESULT_BREAK if break triggered, IPU_BREAK_RESULT_CONTINUE otherwise
 */
ipu__break_result_t ipu__execute_break_instruction(
    ipu__obj_t *ipu, 
    inst_parser__inst_t inst, 
    const ipu__regfile_t *regfile_snapshot);

#endif // IPU_BREAK_INST_H
