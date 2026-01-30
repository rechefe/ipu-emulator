#include "ipu_lr_inst.h"
#include "ipu_regfile.h"
#include "logging/logger.h"
#include <assert.h>
#include <stdbool.h>

// Structure to hold information about a single LR instruction
typedef struct
{
    bool valid;
    inst_parser__lr_inst_opcode_t opcode;
    int lr_idx;
    uint32_t immediate;
    int lcr_a_idx;  // First source register index (for add/sub)
    int lcr_b_idx;  // Second source register index (for add/sub)
} ipu__lr_inst_info_t;

// Maximum number of LR instructions supported per cycle
#define IPU__MAX_LR_INSTS_PER_CYCLE 2

// Helper function to read value from LCR (LR or CR) register
// If index < 16, it's an LR register, otherwise it's a CR register (index - 16)
static inline uint32_t ipu__read_lcr_reg(const ipu__regfile_t *regfile, int lcr_idx)
{
    if (lcr_idx < IPU__LR_REGS_NUM)
    {
        // LR register
        return regfile->lr_regfile.lr[lcr_idx];
    }
    else
    {
        // CR register
        int cr_idx = lcr_idx - IPU__LR_REGS_NUM;
        assert(cr_idx >= 0 && cr_idx < IPU__CR_REGS_NUM);
        return regfile->cr_regfile.cr[cr_idx];
    }
}

// Check if an LR instruction is valid (performs an actual register write)
static inline bool ipu__is_lr_inst_valid(inst_parser__lr_inst_opcode_t opcode, uint32_t immediate)
{
    // INCR by 0 is a NOP and doesn't count as writing to the register
    if (opcode == INST_PARSER__LR_INST_OPCODE_INCR && immediate == 0)
    {
        return false;
    }
    return true;
}

// Extract all LR instructions from an instruction word
static int ipu__extract_lr_instructions(inst_parser__inst_t inst, ipu__lr_inst_info_t *lr_insts)
{
    int count = 0;

    // LR instruction 0
    lr_insts[0].opcode = inst.lr_inst_0_token_0_lr_inst_opcode;
    lr_insts[0].lr_idx = inst.lr_inst_0_token_1_lr_reg_field;
    
    // For add/sub instructions, token_2 and token_3 are lcr registers
    // For set/incr instructions, token_4 is immediate
    if (lr_insts[0].opcode == INST_PARSER__LR_INST_OPCODE_ADD ||
        lr_insts[0].opcode == INST_PARSER__LR_INST_OPCODE_SUB)
    {
        lr_insts[0].lcr_a_idx = inst.lr_inst_0_token_2_lcr_reg_field;
        lr_insts[0].lcr_b_idx = inst.lr_inst_0_token_3_lcr_reg_field;
        lr_insts[0].immediate = 0;
    }
    else
    {
        lr_insts[0].immediate = inst.lr_inst_0_token_4_lr_immediate_type;
        lr_insts[0].lcr_a_idx = 0;
        lr_insts[0].lcr_b_idx = 0;
    }
    lr_insts[0].valid = ipu__is_lr_inst_valid(lr_insts[0].opcode, lr_insts[0].immediate);
    count++;

    // LR instruction 1
    lr_insts[1].opcode = inst.lr_inst_1_token_0_lr_inst_opcode;
    lr_insts[1].lr_idx = inst.lr_inst_1_token_1_lr_reg_field;
    
    if (lr_insts[1].opcode == INST_PARSER__LR_INST_OPCODE_ADD ||
        lr_insts[1].opcode == INST_PARSER__LR_INST_OPCODE_SUB)
    {
        lr_insts[1].lcr_a_idx = inst.lr_inst_1_token_2_lcr_reg_field;
        lr_insts[1].lcr_b_idx = inst.lr_inst_1_token_3_lcr_reg_field;
        lr_insts[1].immediate = 0;
    }
    else
    {
        lr_insts[1].immediate = inst.lr_inst_1_token_4_lr_immediate_type;
        lr_insts[1].lcr_a_idx = 0;
        lr_insts[1].lcr_b_idx = 0;
    }
    lr_insts[1].valid = ipu__is_lr_inst_valid(lr_insts[1].opcode, lr_insts[1].immediate);
    count++;

    // Future expansion: Add more LR instructions here as needed
    // lr_insts[2].opcode = inst.lr_inst_2_token_0_lr_inst_opcode;
    // ...

    return count;
}

// Check for conflicts between LR instructions targeting the same register
static bool ipu__check_lr_conflicts(const ipu__lr_inst_info_t *lr_insts, int count)
{
    // Check if any two instructions write to the same LR register
    for (int i = 0; i < count; i++)
    {
        if (!lr_insts[i].valid)
            continue;

        for (int j = i + 1; j < count; j++)
        {
            if (!lr_insts[j].valid)
                continue;

            if (lr_insts[i].lr_idx == lr_insts[j].lr_idx)
            {
                LOG_ERROR("LR instruction conflict detected: LR%d written by multiple instructions in the same cycle",
                          lr_insts[i].lr_idx);
                return true; // Conflict detected
            }
        }
    }
    return false; // No conflicts
}

// Execute a single LR instruction operation
static void ipu__execute_single_lr_inst(ipu__obj_t *ipu,
                                        const ipu__lr_inst_info_t *lr_inst,
                                        const ipu__regfile_t *regfile_snapshot)
{
    switch (lr_inst->opcode)
    {
    case INST_PARSER__LR_INST_OPCODE_SET:
        // SET LR, immediate
        ipu__set_lr(ipu, lr_inst->lr_idx, lr_inst->immediate);
        break;
    case INST_PARSER__LR_INST_OPCODE_INCR:
        // INCR LR, immediate (add to LR) - read from snapshot
        {
            uint32_t lr_value = regfile_snapshot->lr_regfile.lr[lr_inst->lr_idx];
            ipu->regfile.lr_regfile.lr[lr_inst->lr_idx] = lr_value + lr_inst->immediate;
        }
        break;
    case INST_PARSER__LR_INST_OPCODE_ADD:
        // ADD LRd, LCRa, LCRb
        {
            uint32_t val_a = ipu__read_lcr_reg(regfile_snapshot, lr_inst->lcr_a_idx);
            uint32_t val_b = ipu__read_lcr_reg(regfile_snapshot, lr_inst->lcr_b_idx);
            ipu->regfile.lr_regfile.lr[lr_inst->lr_idx] = val_a + val_b;
        }
        break;
    case INST_PARSER__LR_INST_OPCODE_SUB:
        // SUB LRd, LCRa, LCRb
        {
            uint32_t val_a = ipu__read_lcr_reg(regfile_snapshot, lr_inst->lcr_a_idx);
            uint32_t val_b = ipu__read_lcr_reg(regfile_snapshot, lr_inst->lcr_b_idx);
            ipu->regfile.lr_regfile.lr[lr_inst->lr_idx] = val_a - val_b;
        }
        break;
    default:
        assert(0 && "Unknown LR instruction opcode");
    }
}

void ipu__execute_lr_instruction(ipu__obj_t *ipu, inst_parser__inst_t inst, const ipu__regfile_t *regfile_snapshot)
{
    // Extract all LR instructions from the instruction word
    ipu__lr_inst_info_t lr_insts[IPU__MAX_LR_INSTS_PER_CYCLE];
    int lr_inst_count = ipu__extract_lr_instructions(inst, lr_insts);

    // Check for conflicts between LR instructions
    if (ipu__check_lr_conflicts(lr_insts, lr_inst_count))
    {
        // Conflict detected - halt execution or handle as needed
        assert(0 && "Cannot execute LR instructions with register conflicts");
        return;
    }

    // Execute all LR instructions in parallel (no conflicts)
    for (int i = 0; i < lr_inst_count; i++)
    {
        if (lr_insts[i].valid)
        {
            ipu__execute_single_lr_inst(ipu, &lr_insts[i], regfile_snapshot);
        }
    }
}
