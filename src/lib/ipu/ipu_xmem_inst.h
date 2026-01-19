#ifndef IPU_XMEM_INST_H
#define IPU_XMEM_INST_H

#include "ipu_base.h"

// XMEM instruction execution
void ipu__execute_xmem_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot);

#endif // IPU_XMEM_INST_H
