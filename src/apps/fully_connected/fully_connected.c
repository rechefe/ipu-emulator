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

#define OUTPUT_BASE_ADDR 0x40000
#define OUTPUT_NEURONS 64

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
    LOG_INFO("Loading input activations from: %s", inputs_file);
    FILE *inputs_fp = fopen(inputs_file, "rb");
    if (!inputs_fp)
    {
        LOG_ERROR("Failed to open inputs file: %s", inputs_file);
        return;
    }

    uint8_t input_data[IPU__R_REG_SIZE_BYTES];
    uint32_t input_addr = INPUT_BASE_ADDR;
    size_t inputs_read = 0;
    while (fread(input_data, 1, IPU__R_REG_SIZE_BYTES, inputs_fp) == IPU__R_REG_SIZE_BYTES)
    {
        xmem__write_address(ipu->xmem, input_addr, input_data, IPU__R_REG_SIZE_BYTES);
        input_addr += IPU__R_REG_SIZE_BYTES;
        inputs_read++;
        if (inputs_read >= SAMPLES_NUM)
            break; // Max 256 inputs
    }
    fclose(inputs_fp);
    LOG_INFO("Loaded %zu input activations", inputs_read);

    // Load weights from file
    LOG_INFO("Loading weights from: %s", weights_file);
    FILE *weights_fp = fopen(weights_file, "rb");
    if (!weights_fp)
    {
        LOG_ERROR("Failed to open weights file: %s", weights_file);
        return;
    }

    uint8_t weight_data[IPU__R_REG_SIZE_BYTES];
    uint32_t weight_addr = WEIGHTS_BASE_ADDR;
    size_t weights_read = 0;
    while (fread(weight_data, 1, IPU__R_REG_SIZE_BYTES, weights_fp) == IPU__R_REG_SIZE_BYTES)
    {
        xmem__write_address(ipu->xmem, weight_addr, weight_data, IPU__R_REG_SIZE_BYTES);
        weight_addr += IPU__R_REG_SIZE_BYTES;
        weights_read++;
        if (weights_read >= OUTPUT_NEURONS)
            break; // Max 64 neurons
    }
    fclose(weights_fp);
    LOG_INFO("Loaded %zu weight vectors", weights_read);

    // Inserting the weights base address
    ipu->regfile.cr_regfile.cr[0] = INPUT_BASE_ADDR;   // Input base address
    ipu->regfile.cr_regfile.cr[1] = WEIGHTS_BASE_ADDR; // Weights base address
    ipu->regfile.cr_regfile.cr[2] = OUTPUT_BASE_ADDR;  // Output base address

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

    // Save output activations to file
    LOG_INFO("Saving output activations to: %s", outputs_file);
    FILE *outputs_fp = fopen(outputs_file, "wb");
    if (!outputs_fp)
    {
        LOG_ERROR("Failed to open outputs file: %s", outputs_file);
    }
    else
    {
        uint32_t output_base = OUTPUT_BASE_ADDR;
        uint8_t output_data[IPU__RD_REG_SIZE_BYTES];
        size_t outputs_written = 0;

        // Read and save output activations from memory
        for (int i = 0; i < SAMPLES_NUM; i++) // Up to 256 outputs
        {
            xmem__read_address(ipu->xmem, output_base + (i * IPU__RD_REG_SIZE_BYTES),
                               output_data, IPU__RD_REG_SIZE_BYTES);
            size_t written = fwrite(output_data, 1, IPU__RD_REG_SIZE_BYTES, outputs_fp);
            if (written == IPU__RD_REG_SIZE_BYTES)
            {
                outputs_written++;
            }
        }
        fclose(outputs_fp);
        LOG_INFO("Saved %zu output activations", outputs_written);
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
