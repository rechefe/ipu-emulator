#include "ipu_break_inst.h"
#include "logging/logger.h"

ipu__break_result_t ipu__execute_break_instruction(
    ipu__obj_t *ipu, 
    inst_parser__inst_t inst, 
    const ipu__regfile_t *regfile_snapshot)
{
    switch (inst.break_inst_token_0_break_inst_opcode)
    {
    case INST_PARSER__BREAK_INST_OPCODE_BREAK:
        // Unconditional break - always trigger
        LOG_DEBUG("Break instruction triggered at PC=%u", ipu->program_counter);
        return IPU_BREAK_RESULT_BREAK;

    case INST_PARSER__BREAK_INST_OPCODE_BREAK_IFEQ:
    {
        // Conditional break - check if LR equals immediate
        int lr_idx = inst.break_inst_token_1_lr_reg_field;
        uint32_t lr_value = regfile_snapshot->lr_regfile.lr[lr_idx];
        uint32_t immediate = inst.break_inst_token_2_break_immediate_type;
        
        if (lr_value == immediate)
        {
            LOG_DEBUG("Break.ifeq triggered at PC=%u (lr%d=%u == %u)", 
                     ipu->program_counter, lr_idx, lr_value, immediate);
            return IPU_BREAK_RESULT_BREAK;
        }
        return IPU_BREAK_RESULT_CONTINUE;
    }

    case INST_PARSER__BREAK_INST_OPCODE_BREAK_NOP:
    default:
        // No break - continue execution
        return IPU_BREAK_RESULT_CONTINUE;
    }
}
