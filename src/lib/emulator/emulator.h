#include "ipu/ipu.h"
#include "logging/logger.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Run the IPU until execution completes
 *
 * @param ipu The IPU object to execute
 * @param max_cycles Maximum number of cycles to execute (safety limit)
 * @return Number of cycles executed, or -1 on error
 */
int emulator__run_until_complete(
    ipu__obj_t *ipu,
    uint32_t max_cycles,
    uint32_t cycles_to_print_progress);

/**
 * @brief Configuration for a generic IPU test
 */
typedef struct {
    const char *test_name;              // Name of the test for logging
    uint32_t max_cycles;                 // Maximum cycles to execute
    uint32_t progress_interval;          // How often to print progress
    void (*setup)(ipu__obj_t *ipu, int argc, char **argv);   // Custom setup function
    void (*teardown)(ipu__obj_t *ipu, int argc, char **argv); // Custom teardown function
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