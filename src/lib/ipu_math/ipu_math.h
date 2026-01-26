#ifndef IPU_MATH_H
#define IPU_MATH_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include "fp/fp.h"

// Data type enumeration
typedef enum
{
    IPU_MATH__DTYPE_INT8,
    IPU_MATH__DTYPE_FP8_E4M3,
    IPU_MATH__DTYPE_FP8_E5M2
} ipu_math__dtype_t;

// Generic operations - work on any data type
void ipu_math__mult(const void *a, const void *b, void *result, ipu_math__dtype_t dtype);
void ipu_math__add(const void *a, const void *b, void *result, ipu_math__dtype_t dtype);
void ipu_math__mac(const void *a, const void *b, const void *acc, void *result, ipu_math__dtype_t dtype);

// Vector operations - process arrays of data
void ipu_math__vec_mult(const uint8_t *a, const uint8_t *b, void *result, size_t count, ipu_math__dtype_t dtype);
void ipu_math__vec_add(const uint8_t *a, const uint8_t *b, void *result, size_t count, ipu_math__dtype_t dtype);
void ipu_math__vec_mac(const uint8_t *a, const uint8_t *b, const void *acc, void *result, size_t count, ipu_math__dtype_t dtype);

// Vector-scalar operations
void ipu_math__vec_scalar_mac(const uint8_t *vec, uint8_t scalar, const void *acc, void *result, size_t count, ipu_math__dtype_t dtype);

#endif // IPU_MATH_H
