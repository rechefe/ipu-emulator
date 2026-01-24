#include "ipu_mult_inst.h"
#include "ipu_regfile.h"

void ipu__mult_instruction_mask_and_shift(
    ipu__obj_t *ipu,
    inst_parser__inst_t inst,
    const ipu__regfile_t *regfile_snapshot)
{
    uint32_t lr_mask_idx = regfile_snapshot->lr_regfile.lr[inst.mult_inst_token_3_lr_reg_field];
    int32_t lr_shift = (int32_t)regfile_snapshot->lr_regfile.lr[inst.mult_inst_token_4_lr_reg_field];

    ipu__mask_t mask =
        ipu->regfile.mult_stage_regfile.r_mask.masks
            [lr_mask_idx % (IPU__R_REG_SIZE_BYTES / sizeof(ipu__mask_t))];

    if (lr_shift > 0)
    {
        mask <<= lr_shift;
    }
    else if (lr_shift < 0)
    {
        mask >>= -lr_shift;
    }

    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        if ((mask & 1) == 0)
        {
            ipu->misc.mult_res.words[i] = 0;
        }
        mask >>= 1;
    }
}

void ipu__execute_mult_ev_instruction(ipu__obj_t *ipu,
                                      inst_parser__inst_t inst,
                                      const ipu__regfile_t *regfile_snapshot)
{
    inst_parser__mult_stage_reg_field_t ra_idx =
        (inst_parser__mult_stage_reg_field_t)inst.mult_inst_token_1_mult_stage_reg_field;

    uint32_t lr1_idx = inst.mult_inst_token_2_lr_reg_field;
    uint32_t lr1_val = regfile_snapshot->lr_regfile.lr[lr1_idx];

    uint32_t lr2_idx = inst.mult_inst_token_5_lr_reg_field;
    uint32_t lr2_val = regfile_snapshot->lr_regfile.lr[lr2_idx];

    ipu__r_reg_t *ra_reg_ptr = ipu__get_mult_stage_r_reg(ipu, ra_idx);
    ipu__r_reg_t rb_reg;

    ipu__get_r_cyclic_at_idx(ipu, lr1_val, &rb_reg);

    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        ipu_math__mult(
            &ra_reg_ptr->bytes[lr2_val % IPU__R_REG_SIZE_BYTES],
            &rb_reg.bytes[i],
            &ipu->misc.mult_res.words[i],
            ipu__get_cr_dtype(&ipu->regfile));
    }
}

void ipu__execute_mult_ee_instruction(ipu__obj_t *ipu,
                                      inst_parser__inst_t inst,
                                      const ipu__regfile_t *regfile_snapshot)
{
    inst_parser__mult_stage_reg_field_t ra_idx =
        (inst_parser__mult_stage_reg_field_t)inst.mult_inst_token_1_mult_stage_reg_field;

    uint32_t lr1_idx = inst.mult_inst_token_2_lr_reg_field;
    uint32_t lr1_val = regfile_snapshot->lr_regfile.lr[lr1_idx];

    ipu__r_reg_t *ra_reg_ptr = ipu__get_mult_stage_r_reg(ipu, ra_idx);
    ipu__r_reg_t rb_reg;

    ipu__get_r_cyclic_at_idx(ipu, lr1_val, &rb_reg);

    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        ipu_math__mult(
            &ra_reg_ptr->bytes[i],
            &rb_reg.bytes[i],
            &ipu->misc.mult_res.words[i],
            ipu__get_cr_dtype(&ipu->regfile));
    }
}

void ipu__execute_mult_instruction(ipu__obj_t *ipu,
                                   inst_parser__inst_t inst,
                                   const ipu__regfile_t *regfile_snapshot)
{
    inst_parser__mult_inst_opcode_t opcode =
        (inst_parser__mult_inst_opcode_t)inst.mult_inst_token_0_mult_inst_opcode;

    switch (opcode)
    {
    case INST_PARSER__MULT_INST_OPCODE_MULT_EE:
    {
        ipu__execute_mult_ee_instruction(ipu, inst, regfile_snapshot);
        break;
    }
    case INST_PARSER__MULT_INST_OPCODE_MULT_EV:
    {
        ipu__execute_mult_ev_instruction(ipu, inst, regfile_snapshot);
        break;
    }
    case INST_PARSER__MULT_INST_OPCODE_MULT_NOP:
    {
        // No operation
        break;
    }
    }
}