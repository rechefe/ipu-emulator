#include "ipu_base.h"
#include "xmem/xmem.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

ipu__obj_t *ipu__init_ipu()
{
    ipu__obj_t *ipu = malloc(sizeof(ipu__obj_t));
    memset(&ipu->regfile, 0, sizeof(ipu__regfile_t));
    xmem__obj_t *xmem = xmem__initialize_xmem();
    ipu->xmem = xmem;
    ipu->program_counter = 0;
    memset(ipu->inst_mem, 0, sizeof(ipu->inst_mem)); // Initialize instruction mem
    return ipu;
}

int ipu__get_r_from_r_enum(int r_enum_val)
{
    assert((r_enum_val >= INST_PARSER__RX_REG_FIELD_R0 &&
            r_enum_val <= INST_PARSER__RX_REG_FIELD_R11) ||
           r_enum_val == INST_PARSER__RX_REG_FIELD_MEM_BYPASS);
    return r_enum_val - INST_PARSER__RX_REG_FIELD_R0;
}

int ipu__get_rd_from_r_enum(int r_enum_val)
{
    assert(r_enum_val >= INST_PARSER__RX_REG_FIELD_RD0 &&
           r_enum_val <= INST_PARSER__RX_REG_FIELD_RD10);
    return (r_enum_val - INST_PARSER__RX_REG_FIELD_RD0);
}

int ipu__get_rq_from_r_enum(int r_enum_val)
{
    assert(r_enum_val >= INST_PARSER__RX_REG_FIELD_RQ0 &&
           r_enum_val <= INST_PARSER__RX_REG_FIELD_RQ8);
    return r_enum_val - INST_PARSER__RX_REG_FIELD_RQ0;
}

int ipu__get_ro_from_r_enum(int r_enum_val)
{
    assert(r_enum_val == INST_PARSER__RX_REG_FIELD_RO0);
    return r_enum_val - INST_PARSER__RX_REG_FIELD_RO0;
}
