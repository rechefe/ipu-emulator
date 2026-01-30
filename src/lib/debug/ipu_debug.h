#ifndef IPU_DEBUG_H
#define IPU_DEBUG_H

#include "ipu/ipu_base.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Debug levels for the emulator
 */
typedef enum {
    IPU_DEBUG_LEVEL_0 = 0,  // Print registers to screen
    IPU_DEBUG_LEVEL_1 = 1,  // Also print disassembled instructions
    IPU_DEBUG_LEVEL_2 = 2,  // Save all registers to JSON
} ipu_debug__level_t;

/**
 * @brief Result from the debug prompt
 */
typedef enum {
    IPU_DEBUG_ACTION_CONTINUE = 0,  // Continue execution (skip current break)
    IPU_DEBUG_ACTION_STEP = 1,      // Execute one instruction then break again
    IPU_DEBUG_ACTION_QUIT = 2,      // Quit debugger
} ipu_debug__action_t;

/**
 * @brief Enter the interactive debug prompt
 * 
 * This function is called when a break instruction triggers.
 * It provides an interactive CLI for inspecting and modifying
 * the IPU state.
 * 
 * Debug Levels:
 * - Level 0: Print registers to screen
 * - Level 1: Also print disassembled current instruction
 * - Level 2: Save all registers to JSON file
 * 
 * @param ipu The IPU object to debug
 * @param level The debug level (0-2)
 * @return Action to take after exiting the prompt
 */
ipu_debug__action_t ipu_debug__enter_prompt(ipu__obj_t *ipu, ipu_debug__level_t level);

#ifdef __cplusplus
}
#endif

#endif // IPU_DEBUG_H
