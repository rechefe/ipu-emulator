#include <stdio.h>
#include <stdlib.h>
#include "logging/logger.h"
#include "ipu/ipu.h"

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    logger__init();
    LOG_INFO("Starting ipu-emulator (stub)");

    ipu__obj_t *ipu = ipu__init_ipu();
    if (!ipu) {
        LOG_ERROR("Failed to initialize IPU");
        return 1;
    }

    // TODO: implement emulator CLI

    free(ipu->xmem);
    free(ipu);
    LOG_INFO("ipu-emulator exiting");
    return 0;
}
