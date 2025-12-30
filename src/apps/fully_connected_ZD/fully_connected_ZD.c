#include "logging/logger.h"
#include "emulator/emulator.h"
#include "ipu/ipu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLES_NUM 10

#define INPUT_BASE_ADDR 0x0000
#define INPUT_NEURONS IPU__R_REG_SIZE_BYTES

#define WEIGHTS_BASE_ADDR 0x20000

#define ZEROS_BASE_ADDR 0x30000

#define OUTPUT_BASE_ADDR 0x40000
#define OUTPUT_NEURONS 128

/**
 * @brief Setup function to initialize IPU state for fully connected layer
 */
void ipu_setup(ipu__obj_t *ipu, int argc, char **argv)
{
    if (argc < 4)
    {
        LOG_ERROR("Setup requires: <inst_file> <inputs.bin> <weights.bin> <outputs.bin>");
        return;
    }

    const char *inputs_file = argv[2];
    const char *weights_file = argv[3];
    LOG_INFO("Setting up IPU for fully connected layer...");

    // TODO - this must be done via the IPU itself
    // Clear accumulator registers
    for (int i = 0; i < IPU__RQ_REGS_NUM; i++)
    {
        ipu__clear_rq_reg(ipu, i);
    }

    // Load input activations from file
    int inputs_loaded = emulator__load_binary_to_xmem(
        ipu->xmem, inputs_file, INPUT_BASE_ADDR, IPU__R_REG_SIZE_BYTES, SAMPLES_NUM);
    
    if (inputs_loaded < 0)
    {
        LOG_ERROR("Failed to load inputs");
        return;
    }

    // Load weights from file
    int weights_loaded = emulator__load_binary_to_xmem(
        ipu->xmem, weights_file, WEIGHTS_BASE_ADDR, IPU__R_REG_SIZE_BYTES, OUTPUT_NEURONS);

    if (weights_loaded < 0)
    {
        LOG_ERROR("Failed to load weights");
        return;
    }

    // Load zeros for clearing accumulator registers
    const char *zeros_file = "src/apps/fully_connected_ZD/zeros_512_bytes.bin";
    int zeros_loaded = emulator__load_binary_to_xmem(
        ipu->xmem, zeros_file, ZEROS_BASE_ADDR, IPU__RQ_REG_SIZE_BYTES, 1);

    if (zeros_loaded < 0)
    {
        LOG_ERROR("Failed to load zeros");
        return;
    }

    // Set control register addresses
    ipu->regfile.cr_regfile.cr[0] = INPUT_BASE_ADDR;   // Input base address
    ipu->regfile.cr_regfile.cr[1] = WEIGHTS_BASE_ADDR; // Weights base address
    ipu->regfile.cr_regfile.cr[2] = OUTPUT_BASE_ADDR;  // Output base address
    ipu->regfile.cr_regfile.cr[3] = ZEROS_BASE_ADDR;   // Zeros base address

    LOG_INFO("IPU setup complete.");
}

/**
 * @brief Teardown function to save outputs and cleanup
 */
void ipu_teardown(ipu__obj_t *ipu, int argc, char **argv)
{
    if (argc < 5)
    {
        LOG_ERROR("Teardown requires output filename");
        free(ipu->xmem);
        free(ipu);
        return;
    }

    const char *outputs_file = argv[4];
    LOG_INFO("IPU Teardown - Final State:");
    LOG_INFO("========================================");

    // Display final PC
    LOG_INFO("Final Program Counter: %u", ipu->program_counter);

    // Save output activations to file (128 neurons × 4 bytes = 512 bytes = 1 RQ register)
    int outputs_saved = emulator__dump_xmem_to_binary(
        ipu->xmem, outputs_file, OUTPUT_BASE_ADDR, IPU__RQ_REG_SIZE_BYTES, SAMPLES_NUM);
    
    if (outputs_saved < 0)
    {
        LOG_ERROR("Failed to save outputs");
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
        .test_name = "IPU Fully Connected Layer Example",
        .max_cycles = 1000000,
        .progress_interval = 100,
        .setup = ipu_setup,
        .teardown = ipu_teardown
    };
    
    return emulator__run_test(argc, argv, &config);
}
