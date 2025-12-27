#ifndef IPU_LR_INST_H
#define IPU_LR_INST_H

#include "ipu_base.h"
#include "src/tools/ipu-as-py/inst_parser.h"

// LR instruction execution
void ipu__execute_lr_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot);

#endif // IPU_LR_INST_H
