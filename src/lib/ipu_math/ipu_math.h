#ifndef IPU_MATH_H
#define IPU_MATH_H

#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include "fp/fp.h"

static inline int8_t ipu_math__sign_extend_int4(uint8_t x, bool is_lower_nibble)
{
    uint8_t val = is_lower_nibble ? (x & 0x0F) : ((x >> 4) & 0x0F);
    if (val & 0x08) // Check sign bit
    {
        return (int8_t)(val | 0xF0); // Sign-extend to 8 bits
    }
    else
    {
        return (int8_t)(val & 0x0F); // Positive number
    }
}

static inline int16_t ipu_math__mul_int8(int8_t a, int8_t b)
{
    return (int16_t)a * (int16_t)b;
}

static inline int8_t ipu_math__mul_int4(
    uint8_t a, bool a_is_lower_nibble, uint8_t b, bool b_is_lower_nibble)
{
    int8_t a_sext = ipu_math__sign_extend_int4(a, a_is_lower_nibble);
    int8_t b_sext = ipu_math__sign_extend_int4(b, b_is_lower_nibble);
    return (int8_t)a_sext * (int8_t)b_sext;
}

#endif // IPU_MATH_H
