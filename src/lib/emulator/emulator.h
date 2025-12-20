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