#ifndef IPU_H
#define IPU_H

#include "ipu_base.h"
#include "ipu_regfile.h"
#include "ipu_xmem_inst.h"
#include "ipu_lr_inst.h"
#include "ipu_mult_inst.h"
#include "ipu_acc_inst.h"
#include "ipu_cond_inst.h"
#include <stdio.h>

// Top-level IPU functions
inst_parser__inst_t ipu__fetch_current_instruction(ipu__obj_t *ipu);
void ipu__execute_next_instruction(ipu__obj_t *ipu);

// Instruction memory management
void ipu__load_inst_mem(ipu__obj_t *ipu, FILE *file);

#endif // IPU_H
