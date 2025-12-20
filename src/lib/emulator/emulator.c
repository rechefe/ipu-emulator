#include "emulator.h"

/**
 * @brief Run the IPU until execution completes
 *
 * @param ipu The IPU object to execute
 * @param max_cycles Maximum number of cycles to execute (safety limit)
 * @return Number of cycles executed, or -1 on error
 */
int emulator__run_until_complete(ipu__obj_t *ipu, uint32_t max_cycles, uint32_t cycles_to_print_progress)
{
    LOG_INFO("Starting IPU execution...");

    uint32_t cycle_count = 0;

    while (cycle_count < max_cycles)
    {
        // Check if PC is out of bounds (indicates end of program)
        if (ipu->program_counter >= IPU__INST_MEM_SIZE)
        {
            LOG_INFO("Execution complete: PC out of bounds (halted)");
            break;
        }

        ipu__execute_next_instruction(ipu);
        cycle_count++;

        if (cycle_count % cycles_to_print_progress == 0)
        {
            LOG_INFO("Executed %u cycles, PC=%u", cycle_count, ipu->program_counter);
        }
    }

    if (cycle_count >= max_cycles)
    {
        LOG_WARN("Execution stopped: Maximum cycle limit (%u) reached", max_cycles);
        return -1;
    }

    LOG_INFO("IPU execution finished after %u cycles", cycle_count);
    return cycle_count;
}
