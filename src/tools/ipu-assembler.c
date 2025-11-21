#include <stdio.h>
#include <stdlib.h>
#include "logging/logger.h"

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    logger__init();
    LOG_INFO("Starting ipu-assembler (stub)");

    // TODO: implement assembler

    LOG_INFO("ipu-assembler exiting");
    return 0;
}
