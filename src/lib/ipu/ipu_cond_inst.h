#ifndef IPU_COND_INST_H
#define IPU_COND_INST_H

#include "ipu_base.h"
#include "src/tools/ipu-as-py/inst_parser.h"

// Conditional/branch instruction execution
void ipu__execute_cond_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot);

#endif // IPU_COND_INST_H
