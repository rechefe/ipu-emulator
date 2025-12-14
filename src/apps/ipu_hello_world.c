#include "logging/logger.h"
#include "ipu/ipu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Setup function to initialize IPU state with test data
 *
 * @param ipu The IPU object to configure
 *
 * This function:
 * - Sets initial values for LR and CR registers
 * - Loads test data into R registers
 * - Initializes external memory (XMEM) with sample data
 */
void ipu_setup(ipu__obj_t *ipu)
{
    LOG_INFO("Setting up IPU initial state...");

    // Fill R registers with a test pattern: ascending values
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        ipu->regfile.rx_regfile.r_regs[4].bytes[i] = (uint8_t)(i);
    }

    // Clear accumulator registers
    ipu__clear_rq_reg(ipu, 0);

    // Initialize XMEM with sample data
    LOG_INFO("Initializing external memory (XMEM)...");
    uint8_t sample_data[IPU__R_REG_SIZE_BYTES];

    // Create test pattern: ascending values
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; ++i)
    {
        sample_data[i] = (uint8_t)i;
    }

    // Write data to multiple memory locations
    xmem__write_address(ipu->xmem, 0x2000, sample_data, IPU__R_REG_SIZE_BYTES);

    LOG_INFO("IPU setup complete.");
}

/**
 * @brief Run the IPU until execution completes
 *
 * @param ipu The IPU object to execute
 * @param max_cycles Maximum number of cycles to execute (safety limit)
 * @return Number of cycles executed, or -1 on error
 *
 * Execution stops when:
 * - A breakpoint instruction is encountered
 * - Program counter exceeds instruction memory bounds
 * - Maximum cycle count is reached
 */
int ipu_run_until_complete(ipu__obj_t *ipu, uint32_t max_cycles)
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

        // Execute one instruction cycle
        ipu__execute_next_instruction(ipu);
        cycle_count++;

        // Check if we hit a breakpoint after execution
        if (ipu->program_counter >= IPU__INST_MEM_SIZE)
        {
            LOG_INFO("Execution complete: Breakpoint reached");
            break;
        }

        // Log progress every 100 cycles
        if (cycle_count % 100 == 0)
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

/**
 * @brief Teardown function to print final IPU state and cleanup
 *
 * @param ipu The IPU object to inspect and cleanup
 *
 * Displays:
 * - Final program counter value
 * - Contents of LR and CR registers
 * - Sample of R register values
 * - Sample of RQ accumulator values
 * - Sample of XMEM contents
 */
void ipu_teardown(ipu__obj_t *ipu)
{
    LOG_INFO("IPU Teardown - Final State:");
    LOG_INFO("========================================");

    // Display final PC
    LOG_INFO("Final Program Counter: %u", ipu->program_counter);

    LOG_INFO("RQ[8] Accumulator Contents:");
    for (int i = 0; i < IPU__RQ_REG_SIZE_WORDS; i++)
    {
        LOG_INFO("RQ[8][%d]: %u", i, ipu->regfile.rx_regfile.rq_regs[2].words[i]);
    }

    LOG_INFO("R[0] Register Contents:");
    for (int i = 0; i < IPU__R_REG_SIZE_BYTES; i++)
    {
        LOG_INFO("R[0][%d]: %u", i, ipu->regfile.rx_regfile.r_regs[0].bytes[i]);
    }

    // Display LR registers
    // Cleanup
    LOG_INFO("Cleaning up IPU resources...");
    free(ipu->xmem);
    free(ipu);
    LOG_INFO("IPU resources freed.");
}

int main(int argc, char **argv)
{
    LOG_INFO("IPU Hello World Example Started");
    LOG_INFO("========================================");

    // Check command line arguments
    if (argc < 2)
    {
        LOG_ERROR("Usage: %s <instruction_file.bin>", argv[0]);
        LOG_INFO("Please provide a binary instruction file to load.");
        return 1;
    }

    const char *inst_filename = argv[1];
    LOG_INFO("Loading instructions from: %s", inst_filename);

    // Initialize IPU
    ipu__obj_t *ipu = ipu__init_ipu();
    if (!ipu)
    {
        LOG_ERROR("Failed to initialize IPU.");
        return 1;
    }
    LOG_INFO("IPU initialized successfully.");

    // Load instruction memory from file
    FILE *inst_file = fopen(inst_filename, "rb");
    if (!inst_file)
    {
        LOG_ERROR("Failed to open instruction file: %s", inst_filename);
        free(ipu->xmem);
        free(ipu);
        return 1;
    }

    ipu__load_inst_mem(ipu, inst_file);
    fclose(inst_file);
    LOG_INFO("Instruction memory loaded successfully.");

    // Setup initial state
    ipu_setup(ipu);

    // Run the IPU until completion (with 10000 cycle safety limit)
    int cycles = ipu_run_until_complete(ipu, 10000);

    if (cycles < 0)
    {
        LOG_ERROR("IPU execution failed or exceeded cycle limit.");
        ipu_teardown(ipu);
        return 1;
    }

    LOG_INFO("IPU executed successfully for %d cycles.", cycles);

    // Teardown and display final state
    ipu_teardown(ipu);

    LOG_INFO("========================================");
    LOG_INFO("IPU Hello World Example Finished");
    return 0;
}