#include "ipu/ipu.h"
#include "logging/logger.h"
#include "emulator/emulator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ipu_setup(ipu__obj_t *ipu, int argc, char **argv)
{
    (void)argc; // Unused parameter
    (void)argv; // Unused parameter
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
        sample_data[i] = (uint8_t)i + 10;
    }

    // Write data to multiple memory locations
    xmem__write_address(ipu->xmem, 0x2000, sample_data, IPU__R_REG_SIZE_BYTES);

    LOG_INFO("IPU setup complete.");
}

void ipu_teardown(ipu__obj_t *ipu, int argc, char **argv)
{
    (void)argc; // Unused parameter
    (void)argv; // Unused parameter
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
    LOG_INFO("LR Register Contents:");
    LOG_INFO("LR[1]: 0x%08X (expected: 0x2000)", ipu->regfile.lr_regfile.lr[1]);
    LOG_INFO("LR[2]: 0x%08X (expected: 0x1000)", ipu->regfile.lr_regfile.lr[2]);

    // Verify LR values were set correctly by parallel instruction
    if (ipu->regfile.lr_regfile.lr[1] == 0x2000 && ipu->regfile.lr_regfile.lr[2] == 0x1000)
    {
        LOG_INFO("SUCCESS: Both LR registers set correctly in parallel!");
    }
    else
    {
        LOG_ERROR("FAILURE: LR registers not set correctly!");
    }

    // Cleanup
    LOG_INFO("Cleaning up IPU resources...");
    free(ipu->xmem);
    free(ipu);
    LOG_INFO("IPU resources freed.");
}

int main(int argc, char **argv)
{
    emulator__test_config_t config = {
        .test_name = "IPU Hello World Example",
        .max_cycles = 10000,
        .progress_interval = 100,
        .setup = ipu_setup,
        .teardown = ipu_teardown};

    return emulator__run_test(argc, argv, &config);
}
