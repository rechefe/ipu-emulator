#ifndef EMULATOR_H
#define EMULATOR_H

#include "ipu/ipu.h"
#include "debug/ipu_debug.h"
#include "logging/logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/**
 * @brief Debug configuration for the emulator
 */
typedef struct {
    bool enabled;                   // Whether debug mode is enabled
    ipu_debug__level_t level;       // Debug level (0-2)
} emulator__debug_config_t;

/**
 * @brief Run the IPU until execution completes
 *
 * @param ipu The IPU object to execute
 * @param max_cycles Maximum number of cycles to execute (safety limit)
 * @param cycles_to_print_progress Print progress every N cycles
 * @return Number of cycles executed, or -1 on error
 */
int emulator__run_until_complete(
    ipu__obj_t *ipu,
    uint32_t max_cycles,
    uint32_t cycles_to_print_progress);

/**
 * @brief Run the IPU with debug support
 *
 * @param ipu The IPU object to execute
 * @param max_cycles Maximum number of cycles to execute (safety limit)
 * @param cycles_to_print_progress Print progress every N cycles
 * @param debug_config Debug configuration
 * @return Number of cycles executed, or -1 on error
 */
int emulator__run_with_debug(
    ipu__obj_t *ipu,
    uint32_t max_cycles,
    uint32_t cycles_to_print_progress,
    emulator__debug_config_t *debug_config);

/**
 * @brief Configuration for a generic IPU test
 */
typedef struct {
    const char *test_name;              // Name of the test for logging
    uint32_t max_cycles;                 // Maximum cycles to execute
    uint32_t progress_interval;          // How often to print progress
    void (*setup)(ipu__obj_t *ipu, int argc, char **argv);   // Custom setup function
    void (*teardown)(ipu__obj_t *ipu, int argc, char **argv); // Custom teardown function
    emulator__debug_config_t debug_config; // Debug configuration
} emulator__test_config_t;

/**
 * @brief Run a generic IPU test with minimal boilerplate
 *
 * This function handles:
 * - Argument validation
 * - IPU initialization
 * - Instruction loading
 * - Running until complete
 * - Cleanup
 *
 * @param argc Number of command line arguments
 * @param argv Command line arguments (first must be instruction file)
 * @param config Test configuration structure
 * @return 0 on success, non-zero on failure
 */
int emulator__run_test(int argc, char **argv, emulator__test_config_t *config);

/**
 * @brief Load binary file into XMEM in chunks
 *
 * @param xmem External memory object
 * @param file_path Path to binary file to load
 * @param base_addr Starting address in XMEM
 * @param chunk_size Size of each chunk to read/write
 * @param max_chunks Maximum number of chunks to load (0 = no limit)
 * @return Number of chunks loaded, or -1 on error
 */
int emulator__load_binary_to_xmem(
    xmem__obj_t *xmem,
    const char *file_path,
    uint32_t base_addr,
    size_t chunk_size,
    size_t max_chunks);
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
    uint32_t base_addr);
/**
 * @brief Dump XMEM contents to binary file in chunks
 *
 * @param xmem External memory object
 * @param file_path Path to binary file to write
 * @param base_addr Starting address in XMEM
 * @param chunk_size Size of each chunk to read/write
 * @param num_chunks Number of chunks to dump
 * @return Number of chunks written, or -1 on error
 */
int emulator__dump_xmem_to_binary(
    xmem__obj_t *xmem,
    const char *file_path,
    uint32_t base_addr,
    size_t chunk_size,
    size_t num_chunks);

#endif // EMULATOR_H