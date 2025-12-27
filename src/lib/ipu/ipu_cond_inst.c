#include "ipu_cond_inst.h"
#include "logging/logger.h"
#include <assert.h>

static void ipu__execute_cond_beq(ipu__obj_t *ipu, uint32_t lr1, uint32_t lr2, uint32_t label)
{
    // Branch if equal: if LR1 == LR2 then PC = label
    if (lr1 == lr2)
    {
        ipu->program_counter = label;
    }
    else
    {
        ipu->program_counter += 1;
    }
}

static void ipu__execute_cond_bne(ipu__obj_t *ipu, uint32_t lr1, uint32_t lr2, uint32_t label)
{
    // Branch if not equal: if LR1 != LR2 then PC = label
    if (lr1 != lr2)
    {
        ipu->program_counter = label;
    }
    else
    {
        ipu->program_counter += 1;
    }
}

static void ipu__execute_cond_blt(ipu__obj_t *ipu, uint32_t lr1, uint32_t lr2, uint32_t label)
{
    // Branch if less than: if LR1 < LR2 then PC = label
    if (lr1 < lr2)
    {
        ipu->program_counter = label;
    }
    else
    {
        ipu->program_counter += 1;
    }
}

static void ipu__execute_cond_bnz(ipu__obj_t *ipu, uint32_t lr1, uint32_t label)
{
    // Branch if not zero: if LR1 != 0 then PC = label
    if (lr1 != 0)
    {
        ipu->program_counter = label;
    }
    else
    {
        ipu->program_counter += 1;
    }
}

static void ipu__execute_cond_bz(ipu__obj_t *ipu, uint32_t lr1, uint32_t label)
{
    // Branch if zero: if LR1 == 0 then PC = label
    if (lr1 == 0)
    {
        ipu->program_counter = label;
    }
    else
    {
        ipu->program_counter += 1;
    }
}

static void ipu__execute_cond_b(ipu__obj_t *ipu, uint32_t label)
{
    // Unconditional branch: PC = label
    ipu->program_counter = label;
}

static void ipu__execute_cond_br(ipu__obj_t *ipu, uint32_t lr1)
{
    // Branch register: PC = LR1
    ipu->program_counter = lr1;
}

static void ipu__execute_cond_bkpt(ipu__obj_t *ipu)
{
    // Breakpoint - halt execution
    // Set PC to invalid value to stop
    ipu->program_counter = IPU__INST_MEM_SIZE;
    LOG_INFO("IPU breakpoint reached, halting execution.");
}

void ipu__execute_cond_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    int lr1_idx = inst.cond_inst_token_1_lr_reg_field;
    int lr2_idx = inst.cond_inst_token_2_lr_reg_field;
    // Read LR values from snapshot to avoid race conditions
    uint32_t lr1 = regfile_snapshot->lr_regfile.lr[lr1_idx];
    uint32_t lr2 = regfile_snapshot->lr_regfile.lr[lr2_idx];
    uint32_t label = inst.cond_inst_token_3_label_token;

    switch (inst.cond_inst_token_0_cond_inst_opcode)
    {
    case INST_PARSER__COND_INST_OPCODE_BEQ:
        ipu__execute_cond_beq(ipu, lr1, lr2, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BNE:
        ipu__execute_cond_bne(ipu, lr1, lr2, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BLT:
        ipu__execute_cond_blt(ipu, lr1, lr2, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BNZ:
        ipu__execute_cond_bnz(ipu, lr1, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BZ:
        ipu__execute_cond_bz(ipu, lr1, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_B:
        ipu__execute_cond_b(ipu, label);
        break;
    case INST_PARSER__COND_INST_OPCODE_BR:
        ipu__execute_cond_br(ipu, lr1);
        break;
    case INST_PARSER__COND_INST_OPCODE_BKPT:
        ipu__execute_cond_bkpt(ipu);
        break;
    default:
        assert(0 && "Unknown COND instruction opcode");
    }
}
