#include "emulator.h"
#include <string.h>

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

    // Run the IPU until completion
    int cycles = emulator__run_until_complete(ipu, config->max_cycles, config->progress_interval);

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
