#include "ipu_math/ipu_math.h"
#include <assert.h>
#include <string.h>

// Helper function for sign-extending INT4 values
static int8_t ipu_math__sign_extend_int4(uint8_t value, bool lower_nibble)
{
    uint8_t nibble;
    if (lower_nibble)
    {
        nibble = value & 0x0F; // Extract lower 4 bits
    }
    else
    {
        nibble = (value >> 4) & 0x0F; // Extract upper 4 bits
    }

    // Sign extend from 4 bits to 8 bits
    if (nibble & 0x08) // Check if sign bit is set
    {
        return (int8_t)(nibble | 0xF0); // Sign extend with 1s
    }
    else
    {
        return (int8_t)nibble; // Sign extend with 0s
    }
}

// Generic multiplication function
void ipu_math__mult(const void *a, const void *b, void *result, ipu_math__dtype_t dtype)
{
    switch (dtype)
    {
    case IPU_MATH__DTYPE_INT4_LOWER:
    {
        int8_t a_sext = ipu_math__sign_extend_int4(*(const uint8_t *)a, true);
        int8_t b_sext = ipu_math__sign_extend_int4(*(const uint8_t *)b, true);
        *(int8_t *)result = a_sext * b_sext;
        break;
    }
    case IPU_MATH__DTYPE_INT4_UPPER:
    {
        int8_t a_sext = ipu_math__sign_extend_int4(*(const uint8_t *)a, false);
        int8_t b_sext = ipu_math__sign_extend_int4(*(const uint8_t *)b, false);
        *(int8_t *)result = a_sext * b_sext;
        break;
    }
    case IPU_MATH__DTYPE_INT8:
    {
        int32_t a_val = (int32_t)(*(const int8_t *)a);
        int32_t b_val = (int32_t)(*(const int8_t *)b);
        int32_t res_val = a_val * b_val;
        *(int32_t *)result = res_val;
        break;
    }
    case IPU_MATH__DTYPE_FP4:
        *(float *)result = fp__fp4_mult(*(const fp__fp4_t *)a, *(const fp__fp4_t *)b);
        break;

    case IPU_MATH__DTYPE_FP8_E4M3:
        *(float *)result = fp__fp8_e4m3_mult(*(const fp__fp8_e4m3_t *)a, *(const fp__fp8_e4m3_t *)b);
        break;

    case IPU_MATH__DTYPE_FP8_E5M2:
        *(float *)result = fp__fp8_e5m2_mult(*(const fp__fp8_e5m2_t *)a, *(const fp__fp8_e5m2_t *)b);
        break;

    case IPU_MATH__DTYPE_FP16:
        *(float *)result = fp__fp16_mult(*(const fp__fp16_t *)a, *(const fp__fp16_t *)b);
        break;

    default:
        assert("Unsupported data type for multiplication" && 0);
        break;
    }
}

// Generic addition function
void ipu_math__add(const void *a, const void *b, void *result, ipu_math__dtype_t dtype)
{
    switch (dtype)
    {
    case IPU_MATH__DTYPE_INT4_LOWER:
    {
        int8_t a_sext = ipu_math__sign_extend_int4(*(const uint8_t *)a, true);
        int8_t b_sext = ipu_math__sign_extend_int4(*(const uint8_t *)b, true);
        *(int8_t *)result = a_sext + b_sext;
        break;
    }
    case IPU_MATH__DTYPE_INT4_UPPER:
    {
        int8_t a_sext = ipu_math__sign_extend_int4(*(const uint8_t *)a, false);
        int8_t b_sext = ipu_math__sign_extend_int4(*(const uint8_t *)b, false);
        *(int8_t *)result = a_sext + b_sext;
        break;
    }
    case IPU_MATH__DTYPE_INT8:
        *(int32_t *)result = (int32_t)(*(const int32_t *)a) + (int32_t)(*(const int32_t *)b);
        break;

    case IPU_MATH__DTYPE_FP4:
        *(float *)result = fp__fp4_add(*(const fp__fp4_t *)a, *(const fp__fp4_t *)b);
        break;

    case IPU_MATH__DTYPE_FP8_E4M3:
        *(float *)result = fp__fp8_e4m3_add(*(const fp__fp8_e4m3_t *)a, *(const fp__fp8_e4m3_t *)b);
        break;

    case IPU_MATH__DTYPE_FP8_E5M2:
        *(float *)result = fp__fp8_e5m2_add(*(const fp__fp8_e5m2_t *)a, *(const fp__fp8_e5m2_t *)b);
        break;

    case IPU_MATH__DTYPE_FP16:
        *(float *)result = fp__fp16_add(*(const fp__fp16_t *)a, *(const fp__fp16_t *)b);
        break;

    default:
        assert("Unsupported data type for addition" && 0);
        break;
    }
}

// Generic multiply-accumulate function
void ipu_math__mac(const void *a, const void *b, const void *acc, void *result, ipu_math__dtype_t dtype)
{
    switch (dtype)
    {
    case IPU_MATH__DTYPE_INT4_LOWER:
    case IPU_MATH__DTYPE_INT4_UPPER:
    {
        int8_t mult_result;
        ipu_math__mult(a, b, &mult_result, dtype);
        *(int8_t *)result = *(const int8_t *)acc + mult_result;
        break;
    }
    case IPU_MATH__DTYPE_INT8:
    {
        int32_t mult_result;
        ipu_math__mult(a, b, &mult_result, dtype);
        *(int32_t *)result = *(const int32_t *)acc + mult_result;
        break;
    }
    case IPU_MATH__DTYPE_FP4:
    case IPU_MATH__DTYPE_FP8_E4M3:
    case IPU_MATH__DTYPE_FP8_E5M2:
    case IPU_MATH__DTYPE_FP16:
    {
        float mult_result;
        ipu_math__mult(a, b, &mult_result, dtype);
        *(float *)result = *(const float *)acc + mult_result;
        break;
    }
    default:
        assert("Unsupported data type for MAC operation" && 0);
        break;
    }
}
