#include "ipu_math/ipu_math.h"
#include <assert.h>
#include <string.h>

// Generic multiplication function
void ipu_math__mult(const void *a, const void *b, void *result, ipu_math__dtype_t dtype)
{
    switch (dtype)
    {
    case IPU_MATH__DTYPE_INT8:
    {
        int32_t a_val = (int32_t)(*(const int8_t *)a);
        int32_t b_val = (int32_t)(*(const int8_t *)b);
        int32_t res_val = a_val * b_val;
        *(int32_t *)result = res_val;
        break;
    }

    case IPU_MATH__DTYPE_FP8_E4M3:
        *(float *)result = fp__fp8_e4m3_mult(*(const fp__fp8_e4m3_t *)a, *(const fp__fp8_e4m3_t *)b);
        break;

    case IPU_MATH__DTYPE_FP8_E5M2:
        *(float *)result = fp__fp8_e5m2_mult(*(const fp__fp8_e5m2_t *)a, *(const fp__fp8_e5m2_t *)b);
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
    case IPU_MATH__DTYPE_INT8:
        *(int32_t *)result = (int32_t)(*(const int32_t *)a) + (int32_t)(*(const int32_t *)b);
        break;

    case IPU_MATH__DTYPE_FP8_E4M3:
    case IPU_MATH__DTYPE_FP8_E5M2:
        *(float *)result = *(float *)a + *(float *)b;
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
    case IPU_MATH__DTYPE_INT8:
    {
        int32_t mult_result;
        ipu_math__mult(a, b, &mult_result, dtype);
        *(int32_t *)result = *(const int32_t *)acc + mult_result;
        break;
    }
    case IPU_MATH__DTYPE_FP8_E4M3:
    case IPU_MATH__DTYPE_FP8_E5M2:
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
