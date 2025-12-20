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