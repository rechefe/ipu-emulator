#ifndef IPU_ACC_INST_H
#define IPU_ACC_INST_H

#include "ipu_base.h"

void ipu__execute_acc_instruction(ipu__obj_t *ipu,
                                  inst_parser__inst_t inst,
                                  const ipu__regfile_t *regfile_snapshot);

#endif // IPU_ACC_INST_H
