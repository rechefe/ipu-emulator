#include "ipu_xmem_inst.h"
#include "ipu_regfile.h"
#include "xmem/xmem.h"
#include <assert.h>

static void ipu__execute_xmem_str(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // Store R register to external memory: STR RX, [LR, CR]
    int rx_idx = ipu__get_r_from_r_enum(inst.xmem_inst_token_1_rx_reg_field);

    // Currently stores R0 to XMEM at address LR + CR
    int lr_idx = inst.xmem_inst_token_2_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_3_cr_reg_field;

    if (rx_idx != INST_PARSER__RX_REG_FIELD_MEM_BYPASS)
    {
        ipu__store_r_reg(ipu, rx_idx, cr_idx, lr_idx, regfile_snapshot);
    }
    else
    {
        assert(0 && "Storing from MEM_BYPASS register is not supported");
    }
}

static void ipu__execute_xmem_ldr(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // Load R register from external memory: LDR RX, [LR, CR]
    int rx_idx = ipu__get_r_from_r_enum(inst.xmem_inst_token_1_rx_reg_field);

    // Currently loads into R0 from XMEM at address LR + CR
    int lr_idx = inst.xmem_inst_token_2_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_3_cr_reg_field;
    if (rx_idx != INST_PARSER__RX_REG_FIELD_MEM_BYPASS)
    {
        ipu__load_r_reg(ipu, rx_idx, cr_idx, lr_idx, regfile_snapshot);
    }
    else
    {
        xmem__read_address(ipu->xmem,
                           regfile_snapshot->lr_regfile.lr[lr_idx] + regfile_snapshot->cr_regfile.cr[cr_idx],
                           (uint8_t *)&ipu->misc.mem_bypass_reg,
                           IPU__R_REG_SIZE_BYTES);
    }
}

void ipu__execute_xmem_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    switch (inst.xmem_inst_token_0_xmem_inst_opcode)
    {
    case INST_PARSER__XMEM_INST_OPCODE_STR:
        ipu__execute_xmem_str(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__XMEM_INST_OPCODE_LDR:
        ipu__execute_xmem_ldr(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__XMEM_INST_OPCODE_XMEM_NOP:
        // No operation for XMEM
        break;
    default:
        assert(0 && "Unknown XMEM instruction opcode");
    }
}
