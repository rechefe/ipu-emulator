#include "emulator.h"
#include "debug/ipu_debug.h"
#include "fp/fp.h"
#include "ipu/ipu.h"
#include <string.h>
#include <stdlib.h>

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

        ipu__break_result_t result = ipu__execute_next_instruction(ipu);
        (void)result; // Ignore break result in non-debug mode
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

/**
 * @brief Run the IPU with debug support
 */
int emulator__run_with_debug(
    ipu__obj_t *ipu,
    uint32_t max_cycles,
    uint32_t cycles_to_print_progress,
    emulator__debug_config_t *debug_config)
{
    LOG_INFO("Starting IPU execution with debug mode...");

    uint32_t cycle_count = 0;
    bool step_mode = false;  // When true, break after each instruction

    while (cycle_count < max_cycles)
    {
        // Check if PC is out of bounds (indicates end of program)
        if (ipu->program_counter >= IPU__INST_MEM_SIZE)
        {
            LOG_INFO("Execution complete: PC out of bounds (halted)");
            break;
        }

        ipu__break_result_t result = ipu__execute_next_instruction(ipu);
        
        // Check for break or step mode
        if ((result == IPU_BREAK_RESULT_BREAK || step_mode) && debug_config->enabled)
        {
            if (result == IPU_BREAK_RESULT_BREAK)
            {
                LOG_INFO("Break triggered at PC=%u, entering debug prompt...", ipu->program_counter);
            }
            else
            {
                LOG_INFO("Step complete at PC=%u", ipu->program_counter);
            }
            
            ipu_debug__action_t action = ipu_debug__enter_prompt(ipu, debug_config->level);
            
            switch (action)
            {
                case IPU_DEBUG_ACTION_CONTINUE:
                    step_mode = false;
                    if (result == IPU_BREAK_RESULT_BREAK)
                    {
                        // We hit a break and didn't execute the rest of the instruction
                        // Execute it now, skipping the break check
                        ipu__execute_instruction_skip_break(ipu);
                    }
                    break;
                    
                case IPU_DEBUG_ACTION_STEP:
                    step_mode = true;
                    if (result == IPU_BREAK_RESULT_BREAK)
                    {
                        // Execute current instruction (skipping break)
                        ipu__execute_instruction_skip_break(ipu);
                    }
                    break;
                    
                case IPU_DEBUG_ACTION_QUIT:
                    LOG_INFO("Debug quit - halting execution");
                    return cycle_count;
            }
        }
        
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

/**
 * @brief Run a generic IPU test with minimal boilerplate
 */
int emulator__run_test(int argc, char **argv, emulator__test_config_t *config)
{
    LOG_INFO("%s Started", config->test_name);
    LOG_INFO("========================================");

    // Check minimum arguments (instruction file is always required)
    if (argc < 2)
    {
        LOG_ERROR("Usage: %s <instruction_file.bin> [additional args...]", argv[0]);
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

    // Call custom setup function if provided
    if (config->setup)
    {
        config->setup(ipu, argc, argv);
    }

    // Run the IPU until completion (with debug if enabled)
    int cycles;
    if (config->debug_config.enabled)
    {
        emulator__debug_config_t debug_cfg = config->debug_config;
        cycles = emulator__run_with_debug(ipu, config->max_cycles, config->progress_interval, &debug_cfg);
    }
    else
    {
        cycles = emulator__run_until_complete(ipu, config->max_cycles, config->progress_interval);
    }

    if (cycles < 0)
    {
        LOG_ERROR("IPU execution failed or exceeded cycle limit.");
        if (config->teardown)
        {
            config->teardown(ipu, argc, argv);
        }
        else
        {
            free(ipu->xmem);
            free(ipu);
        }
        return 1;
    }

    LOG_INFO("IPU executed successfully for %d cycles.", cycles);

    // Call custom teardown function if provided
    if (config->teardown)
    {
        config->teardown(ipu, argc, argv);
    }
    else
    {
        // Default cleanup
        free(ipu->xmem);
        free(ipu);
    }

    LOG_INFO("========================================");
    LOG_INFO("%s Finished", config->test_name);
    return 0;
}

/**
 * @brief Load binary file into XMEM in chunks
 */
int emulator__load_binary_to_xmem(
    xmem__obj_t *xmem,
    const char *file_path,
    uint32_t base_addr,
    size_t chunk_size,
    size_t max_chunks)
{
    LOG_INFO("Loading binary file to XMEM: %s", file_path);

    FILE *fp = fopen(file_path, "rb");
    if (!fp)
    {
        LOG_ERROR("Failed to open file: %s", file_path);
        return -1;
    }

    uint8_t *buffer = (uint8_t *)malloc(chunk_size);
    if (!buffer)
    {
        LOG_ERROR("Failed to allocate buffer of size %zu", chunk_size);
        fclose(fp);
        return -1;
    }

    uint32_t addr = base_addr;
    size_t chunks_loaded = 0;

    while (fread(buffer, 1, chunk_size, fp) == chunk_size)
    {
        xmem__write_address(xmem, addr, buffer, chunk_size);
        addr += chunk_size;
        chunks_loaded++;

        if (max_chunks > 0 && chunks_loaded >= max_chunks)
        {
            break;
        }
    }

    free(buffer);
    fclose(fp);

    LOG_INFO("Loaded %zu chunks of %zu bytes each to XMEM starting at 0x%08X",
             chunks_loaded, chunk_size, base_addr);

    return chunks_loaded;
}

/**
 * @brief Dump XMEM contents to binary file in chunks
 */
int emulator__dump_xmem_to_binary(
    xmem__obj_t *xmem,
    const char *file_path,
    uint32_t base_addr,
    size_t chunk_size,
    size_t num_chunks)
{
    LOG_INFO("Dumping XMEM to binary file: %s", file_path);

    FILE *fp = fopen(file_path, "wb");
    if (!fp)
    {
        LOG_ERROR("Failed to open file for writing: %s", file_path);
        return -1;
    }

    uint8_t *buffer = (uint8_t *)malloc(chunk_size);
    if (!buffer)
    {
        LOG_ERROR("Failed to allocate buffer of size %zu", chunk_size);
        fclose(fp);
        return -1;
    }

    uint32_t addr = base_addr;
    size_t chunks_written = 0;

    for (size_t i = 0; i < num_chunks; i++)
    {
        xmem__read_address(xmem, addr, buffer, chunk_size);

        if (fwrite(buffer, 1, chunk_size, fp) != chunk_size)
        {
            LOG_ERROR("Failed to write chunk %zu to file", i);
            break;
        }

        addr += chunk_size;
        chunks_written++;
    }

    free(buffer);
    fclose(fp);

    LOG_INFO("Dumped %zu chunks of %zu bytes each from XMEM starting at 0x%08X",
             chunks_written, chunk_size, base_addr);

    return chunks_written;
}
/**
 * @brief Load FP32 binary file and convert to FP8 E4M3, then store in XMEM
 *
 * Reads a binary file containing float32 values, converts each to FP8 E4M3 format,
 * and stores the converted bytes into XMEM.
 *
 * @param xmem External memory object
 * @param file_path Path to binary file containing FP32 values
 * @param base_addr Starting address in XMEM to write converted data
 * @return Number of FP32 values converted and loaded, or -1 on error
 */
int emulator__load_fp32_as_fp8_e4m3_to_xmem(
    xmem__obj_t *xmem,
    const char *file_path,
    uint32_t base_addr)
{
    LOG_INFO("Loading FP32 binary file and converting to FP8 E4M3: %s", file_path);
    
    FILE *fp = fopen(file_path, "rb");
    if (!fp)
    {
        LOG_ERROR("Failed to open file: %s", file_path);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    int num_fp32_values = file_size / sizeof(float);
    if (file_size % sizeof(float) != 0)
    {
        LOG_WARN("File size %ld is not a multiple of sizeof(float), will read %d values", 
                 file_size, num_fp32_values);
    }

    float *fp32_buffer = (float *)malloc(file_size);
    if (!fp32_buffer)
    {
        LOG_ERROR("Failed to allocate buffer of size %ld for FP32 values", file_size);
        fclose(fp);
        return -1;
    }

    size_t bytes_read = fread(fp32_buffer, 1, file_size, fp);
    fclose(fp);

    if (bytes_read != (size_t)file_size)
    {
        LOG_ERROR("Failed to read complete file. Expected %ld bytes, got %zu", 
                  file_size, bytes_read);
        free(fp32_buffer);
        return -1;
    }

    fp__fp8_e4m3_t *fp8_buffer = (fp__fp8_e4m3_t *)malloc(num_fp32_values * sizeof(fp__fp8_e4m3_t));
    if (!fp8_buffer)
    {
        LOG_ERROR("Failed to allocate buffer for FP8 E4M3 values");
        free(fp32_buffer);
        return -1;
    }

    for (int i = 0; i < num_fp32_values; i++)
    {
        fp8_buffer[i] = fp__fp32_to_fp8_e4m3(fp32_buffer[i]);
    }
    LOG_INFO("Converted %d FP32 values to FP8 E4M3", num_fp32_values);

    uint8_t *fp8_bytes = (uint8_t *)fp8_buffer;
    size_t fp8_size = num_fp32_values * sizeof(fp__fp8_e4m3_t);
    xmem__write_address(xmem, base_addr, fp8_bytes, fp8_size);
    
    LOG_INFO("Loaded %d FP8 E4M3 values to XMEM starting at 0x%08X", 
             num_fp32_values, base_addr);

    free(fp32_buffer);
    free(fp8_buffer);

    return num_fp32_values;
}