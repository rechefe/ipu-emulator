#include "ipu_xmem_inst.h"
#include "ipu_regfile.h"
#include "xmem/xmem.h"
#include <assert.h>

static void ipu__execute_xmem_str_acc_reg(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // Currently stores ACC stage to XMEM at address LR + CR
    int lr_idx = inst.xmem_inst_token_2_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_4_cr_reg_field;

    uint32_t lr_value = regfile_snapshot->lr_regfile.lr[lr_idx];
    uint32_t cr_value = regfile_snapshot->cr_regfile.cr[cr_idx];
    uint32_t target_address = lr_value + cr_value;

    xmem__load_array_to(ipu->xmem, ipu->regfile.acc_stage_regfile.r_acc.bytes,
                        IPU__R_ACC_REG_SIZE_BYTES, target_address);
}

static void ipu__execute_xmem_ldr_mult_reg(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    int lr_idx = inst.xmem_inst_token_2_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_4_cr_reg_field;

    uint32_t lr_value = regfile_snapshot->lr_regfile.lr[lr_idx];
    uint32_t cr_value = regfile_snapshot->cr_regfile.cr[cr_idx];
    uint32_t source_address = lr_value + cr_value;

    ipu__r_reg_t *r_reg_to_load_to;
    inst_parser__mult_stage_reg_field_t mult_stage_idx =
        (inst_parser__mult_stage_reg_field_t)inst.xmem_inst_token_1_mult_stage_reg_field;
    switch (mult_stage_idx)
    {
    case INST_PARSER__MULT_STAGE_REG_FIELD_R0:
        r_reg_to_load_to = &ipu->regfile.mult_stage_regfile.r_regs[0];
        break;
    case INST_PARSER__MULT_STAGE_REG_FIELD_R1:
        r_reg_to_load_to = &ipu->regfile.mult_stage_regfile.r_regs[1];
        break;
    case INST_PARSER__MULT_STAGE_REG_FIELD_MEM_BYPASS:
        r_reg_to_load_to = &ipu->misc.mem_bypass_r_reg;
        break;
    }

    xmem__read_address(ipu->xmem, source_address, r_reg_to_load_to->bytes, IPU__R_REG_SIZE_BYTES);
}

static void ipu__execute_xmem_ldr_mult_mask_reg(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    int lr_idx = inst.xmem_inst_token_2_lr_reg_field;
    int cr_idx = inst.xmem_inst_token_4_cr_reg_field;

    uint32_t lr_value = regfile_snapshot->lr_regfile.lr[lr_idx];
    uint32_t cr_value = regfile_snapshot->cr_regfile.cr[cr_idx];
    uint32_t source_address = lr_value + cr_value;

    xmem__read_address(ipu->xmem, source_address,
                       ipu->regfile.mult_stage_regfile.r_mask.bytes,
                       IPU__R_REG_SIZE_BYTES);
}

static void ipu__execute_xmem_ldr_cyclic_mult_reg(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    int lr_offset_idx = inst.xmem_inst_token_2_lr_reg_field;
    int cr_offset_idx = inst.xmem_inst_token_4_cr_reg_field;

    uint32_t lr_offset_value = regfile_snapshot->lr_regfile.lr[lr_offset_idx];
    uint32_t cr_offset_value = regfile_snapshot->cr_regfile.cr[cr_offset_idx];
    uint32_t source_address = lr_offset_value + cr_offset_value;

    ipu__r_reg_t loaded_reg;
    xmem__read_address(ipu->xmem, source_address, loaded_reg.bytes, IPU__R_REG_SIZE_BYTES);

    int lr_idx_idx = inst.xmem_inst_token_3_lr_reg_field;
    uint32_t lr_idx_value = regfile_snapshot->lr_regfile.lr[lr_idx_idx];

    assert(lr_idx_value % IPU__R_REG_SIZE_BYTES == 0 && "LR index must be aligned to R register size");

    ipu__set_r_cyclic_at_idx(ipu, lr_idx_value, &loaded_reg);
}

void ipu__execute_xmem_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    switch (inst.xmem_inst_token_0_xmem_inst_opcode)
    {
    case INST_PARSER__XMEM_INST_OPCODE_STR_ACC_REG:
        ipu__execute_xmem_str_acc_reg(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__XMEM_INST_OPCODE_LDR_MULT_REG:
        ipu__execute_xmem_ldr_mult_reg(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__XMEM_INST_OPCODE_LDR_CYCLIC_MULT_REG:
        ipu__execute_xmem_ldr_cyclic_mult_reg(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__XMEM_INST_OPCODE_LDR_MULT_MASK_REG:
        ipu__execute_xmem_ldr_mult_mask_reg(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__XMEM_INST_OPCODE_XMEM_NOP:
        // No operation for XMEM
        break;
    default:
        assert(0 && "Unknown XMEM instruction opcode");
    }
}
