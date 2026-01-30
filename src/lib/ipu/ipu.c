#include "ipu.h"
#include <string.h>
#include <assert.h>

inst_parser__inst_t ipu__fetch_current_instruction(ipu__obj_t *ipu)
{
    return ipu->inst_mem[ipu->program_counter];
}

ipu__break_result_t ipu__execute_next_instruction(ipu__obj_t *ipu)
{
    // Fetch instruction once at the start of the cycle
    inst_parser__inst_t inst = ipu__fetch_current_instruction(ipu);

    // Create a snapshot of the register file at the start of the cycle
    // All subinstructions read from this snapshot to avoid race conditions
    ipu__regfile_t regfile_snapshot = ipu->regfile;

    // Execute break instruction FIRST - before any side effects
    ipu__break_result_t break_result = ipu__execute_break_instruction(ipu, inst, &regfile_snapshot);
    if (break_result == IPU_BREAK_RESULT_BREAK)
    {
        // Break triggered - halt before executing other instructions
        return IPU_BREAK_RESULT_BREAK;
    }

    // Execute all subinstructions in parallel using the snapshot and fetched instruction
    ipu__execute_xmem_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_lr_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_mult_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_acc_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_cond_instruction(ipu, inst, &regfile_snapshot);

    return IPU_BREAK_RESULT_CONTINUE;
}

void ipu__execute_instruction_skip_break(ipu__obj_t *ipu)
{
    // Fetch instruction once at the start of the cycle
    inst_parser__inst_t inst = ipu__fetch_current_instruction(ipu);

    // Create a snapshot of the register file at the start of the cycle
    ipu__regfile_t regfile_snapshot = ipu->regfile;

    // Execute all subinstructions EXCEPT break
    ipu__execute_xmem_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_lr_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_mult_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_acc_instruction(ipu, inst, &regfile_snapshot);
    ipu__execute_cond_instruction(ipu, inst, &regfile_snapshot);
}

void ipu__load_inst_mem(ipu__obj_t *ipu, FILE *file)
{
    size_t inst_size = sizeof(inst_parser__inst_t);
    size_t i;
    for (i = 0; i < IPU__INST_MEM_SIZE; i++)
    {
        size_t bytes_read = fread(&ipu->inst_mem[i], 1, inst_size, file);
        if (bytes_read == 0)
        {
            // Reached end of file
            break;
        }
        if (bytes_read != inst_size)
        {
            // Partial read - error
            assert(0 && "Partial instruction read from file");
        }
    }
    // Fill remaining instructions with zeros (NOP)
    for (; i < IPU__INST_MEM_SIZE; i++)
    {
        memset(&ipu->inst_mem[i], 0, inst_size);
    }
}