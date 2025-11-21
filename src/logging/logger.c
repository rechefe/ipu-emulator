#include "logging/logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

static log_level_t current_level = LOG_LEVEL_INFO;

void logger__init()
{
    const char *env = getenv("LOG_LEVEL");
    if (!env) return;

    if (strcmp(env, "DEBUG") == 0) current_level = LOG_LEVEL_DEBUG;
    else if (strcmp(env, "INFO") == 0) current_level = LOG_LEVEL_INFO;
    else if (strcmp(env, "WARN") == 0) current_level = LOG_LEVEL_WARN;
    else if (strcmp(env, "ERROR") == 0) current_level = LOG_LEVEL_ERROR;
    else if (strcmp(env, "OFF") == 0) current_level = LOG_LEVEL_OFF;
}

void logger__set_level(log_level_t level)
{
    current_level = level;
}

log_level_t logger__get_level()
{
    return current_level;
}

void logger__log(log_level_t level, const char *fmt, ...)
{
    (void)level; // level already filtered by macros
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
