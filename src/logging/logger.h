#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <stdarg.h>

typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_OFF
} log_level_t;

/* Initialize the logger (reads env var LOG_LEVEL optionally).
 * Call before using logging macros if you need env-based config.
 */
void logger__init();

/* Set runtime log level. Messages below this level are ignored. */
void logger__set_level(log_level_t level);

/* Get current level */
log_level_t logger__get_level();

/* Low-level logging function used by macros. */
void logger__log(log_level_t level, const char *fmt, ...);

/* Convenient macros */
#define LOG_DEBUG(fmt, ...) do { if (logger__get_level() <= LOG_LEVEL_DEBUG) logger__log(LOG_LEVEL_DEBUG, "DEBUG: " fmt, ##__VA_ARGS__); } while(0)
#define LOG_INFO(fmt, ...)  do { if (logger__get_level() <= LOG_LEVEL_INFO)  logger__log(LOG_LEVEL_INFO,  "INFO: "  fmt, ##__VA_ARGS__); } while(0)
#define LOG_WARN(fmt, ...)  do { if (logger__get_level() <= LOG_LEVEL_WARN)  logger__log(LOG_LEVEL_WARN,  "WARN: "  fmt, ##__VA_ARGS__); } while(0)
#define LOG_ERROR(fmt, ...) do { if (logger__get_level() <= LOG_LEVEL_ERROR) logger__log(LOG_LEVEL_ERROR, "ERROR: " fmt, ##__VA_ARGS__); } while(0)

#endif // LOGGER_H
