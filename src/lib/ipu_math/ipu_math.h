#ifndef IPU_MATH_H
#define IPU_MATH_H

#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include "fp/fp.h"

// Data type enumeration
typedef enum
{
    IPU_MATH__DTYPE_INT4_LOWER, // INT4 in lower nibble
    IPU_MATH__DTYPE_INT4_UPPER, // INT4 in upper nibble
    IPU_MATH__DTYPE_INT8,
    IPU_MATH__DTYPE_FP4,
    IPU_MATH__DTYPE_FP8_E4M3,
    IPU_MATH__DTYPE_FP8_E5M2,
    IPU_MATH__DTYPE_FP16
} ipu_math__dtype_t;

// Generic operations - work on any data type
void ipu_math__mult(const void *a, const void *b, void *result, ipu_math__dtype_t dtype);
void ipu_math__add(const void *a, const void *b, void *result, ipu_math__dtype_t dtype);
void ipu_math__mac(const void *a, const void *b, const void *acc, void *result, ipu_math__dtype_t dtype);

#endif // IPU_MATH_H
