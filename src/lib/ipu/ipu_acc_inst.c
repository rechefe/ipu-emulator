#include "ipu_acc_inst.h"

void ipu__execute_acc_acc_instruction(ipu__obj_t *ipu,
                                      inst_parser__inst_t inst,
                                      const ipu__regfile_t *regfile_snapshot)
{
    (void)inst;

    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        ipu_math__add(
            &regfile_snapshot->acc_stage_regfile.r_acc.words[i],
            &ipu->misc.mult_res.words[i],
            &ipu->regfile.acc_stage_regfile.r_acc.words[i],
            ipu__get_cr_dtype(&ipu->regfile));
    }
}

void ipu__execute_reset_acc_instruction(ipu__obj_t *ipu,
                                         inst_parser__inst_t inst,
                                         const ipu__regfile_t *regfile_snapshot)
{
    (void)inst;
    (void)regfile_snapshot;

    for (int i = 0; i < IPU__R_ACC_REG_SIZE_BYTES; i++)
    {
        ipu->regfile.acc_stage_regfile.r_acc.bytes[i] = 0;
    }
}

void ipu__execute_acc_instruction(ipu__obj_t *ipu,
                                  inst_parser__inst_t inst,
                                  const ipu__regfile_t *regfile_snapshot)
{
    switch (inst.acc_inst_token_0_acc_inst_opcode)
    {
    case INST_PARSER__ACC_INST_OPCODE_ACC:
        ipu__execute_acc_acc_instruction(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__ACC_INST_OPCODE_RESET_ACC:
        ipu__execute_reset_acc_instruction(ipu, inst, regfile_snapshot);
        break;
    case INST_PARSER__ACC_INST_OPCODE_ACC_NOP:
        // No operation
        break;
    default:
        assert(false && "Invalid ACC instruction type");
        break;
    }
}