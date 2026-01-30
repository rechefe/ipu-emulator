#include "logging/logger.h"
#include "emulator/emulator.h"
#include "ipu/ipu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define SAMPLES_NUM 10

#define INPUT_BASE_ADDR 0x0000
#define INPUT_NEURONS IPU__R_REG_SIZE_BYTES

#define WEIGHTS_BASE_ADDR 0x20000

#define OUTPUT_BASE_ADDR 0x40000
#define OUTPUT_NEURONS 64

/**
 * @brief Parse dtype from command line argument
 */
ipu_math__dtype_t parse_dtype(const char *dtype_str)
{
    if (strcmp(dtype_str, "INT8") == 0 || strcmp(dtype_str, "int8") == 0)
    {
        return IPU_MATH__DTYPE_INT8;
    }
    else if (strcmp(dtype_str, "FP8_E4M3") == 0 || strcmp(dtype_str, "fp8_e4m3") == 0)
    {
        return IPU_MATH__DTYPE_FP8_E4M3;
    }
    else if (strcmp(dtype_str, "FP8_E5M2") == 0 || strcmp(dtype_str, "fp8_e5m2") == 0)
    {
        return IPU_MATH__DTYPE_FP8_E5M2;
    }
    else
    {
        LOG_ERROR("Invalid dtype '%s'. Supported: INT8, FP8_E4M3, FP8_E5M2", dtype_str);
        exit(-1);
    }
}

/**
 * @brief Setup function to initialize IPU state for fully connected layer
 */
void ipu_setup(ipu__obj_t *ipu, int argc, char **argv)
{
    if (argc < 6)
    {
        LOG_ERROR("Usage: %s <inst_file> <inputs.bin> <weights.bin> <outputs.bin> <dtype>", argv[0]);
        LOG_ERROR("  dtype: INT8, FP8_E4M3, or FP8_E5M2");
        return;
    }

    const char *inputs_file = argv[2];
    const char *weights_file = argv[3];
    const char *dtype_str = argv[5];

    // Parse and set data type
    ipu_math__dtype_t dtype = parse_dtype(dtype_str);
    ipu__set_cr_dtype(ipu, dtype);
    LOG_INFO("Setting up IPU for fully connected layer with dtype: %s", dtype_str);

    // Load input activations from file (raw 8-bit data)
    int inputs_loaded = emulator__load_binary_to_xmem(
        ipu->xmem, inputs_file, INPUT_BASE_ADDR, IPU__R_REG_SIZE_BYTES, SAMPLES_NUM);

    if (inputs_loaded < 0)
    {
        LOG_ERROR("Failed to load inputs");
        return;
    }
    LOG_INFO("Loaded %d input samples", inputs_loaded);

    // Load and transpose weights from file
    // Original format: 64 vectors of 128 bytes (64x128)
    // New format: 128 vectors of 128 bytes with first 64 bytes from transposed data, rest padded with zeros
    FILE *weights_fp = fopen(weights_file, "rb");
    if (!weights_fp)
    {
        LOG_ERROR("Failed to open weights file: %s", weights_file);
        return;
    }

    // Read original weights (64 vectors of 128 bytes)
    uint8_t original_weights[OUTPUT_NEURONS][INPUT_NEURONS];
    size_t read_count = fread(original_weights, 1, OUTPUT_NEURONS * INPUT_NEURONS, weights_fp);
    fclose(weights_fp);

    if (read_count != OUTPUT_NEURONS * INPUT_NEURONS)
    {
        LOG_ERROR("Failed to read weights file (expected %zu bytes, got %zu)", 
                  OUTPUT_NEURONS * INPUT_NEURONS, read_count);
        return;
    }

    // Transpose and pad: Create 128 vectors of 128 bytes
    // Each new vector contains one element from each of the 64 original vectors, padded with zeros
    for (int i = 0; i < INPUT_NEURONS; i++)
    {
        uint8_t transposed_vector[INPUT_NEURONS];
        memset(transposed_vector, 0, INPUT_NEURONS); // Initialize with zeros

        // Copy the i-th element from each of the 64 original vectors
        for (int j = 0; j < OUTPUT_NEURONS; j++)
        {
            transposed_vector[j] = original_weights[j][i];
        }

        // Write to xmem at appropriate offset
        uint32_t offset = WEIGHTS_BASE_ADDR + (i * INPUT_NEURONS);
        xmem__write_address(ipu->xmem, offset, transposed_vector, INPUT_NEURONS);
    }

    LOG_INFO("Loaded and transposed %d weight vectors (128 vectors of 128 bytes)", INPUT_NEURONS);

    // Set control register addresses
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
    if (argc < 6)
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
    int outputs_saved = emulator__dump_xmem_to_binary(
        ipu->xmem, outputs_file, OUTPUT_BASE_ADDR, OUTPUT_NEURONS * sizeof(uint32_t), SAMPLES_NUM);

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
    // Check for --debug flag
    bool debug_enabled = false;
    int debug_level = 1;  // Default to level 1 (with disassembly)
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0) {
            debug_enabled = true;
        } else if (strncmp(argv[i], "--debug-level=", 14) == 0) {
            debug_level = atoi(argv[i] + 14);
            debug_enabled = true;
        }
    }

    emulator__test_config_t config = {
        .test_name = "IPU Fully Connected Layer Example",
        .max_cycles = 1000000,
        .progress_interval = 100,
        .setup = ipu_setup,
        .teardown = ipu_teardown,
        .debug_config = {
            .enabled = debug_enabled,
            .level = (ipu_debug__level_t)debug_level
        }
    };

    return emulator__run_test(argc, argv, &config);
}
