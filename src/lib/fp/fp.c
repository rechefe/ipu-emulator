#include "fp/fp.h"
#include "xmem/xmem.h"
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Generic conversion function that handles all floating-point types
float fp__convert_to_fp32(uint8_t sign, uint32_t exp, uint32_t man,
                          int exp_bits, int man_bits)
{
    // Handle zero: exp=0, man=0 always represents zero in IEEE754 style
    if (exp == 0 && man == 0)
        return 0.0f;

    // Calculate bias: 2^(exp_bits-1) - 1
    int exp_bias = (1 << (exp_bits - 1)) - 1;
    int fp32_bias = FP__FP32_BIAS;

    // Convert exponent
    int converted_exp = (int)exp - exp_bias + fp32_bias; // Normal number path

    // For true subnormal numbers (when encoding had exp < 0, forced to 0),
    // the mantissa was extracted with one extra bit shift during encoding
    uint32_t converted_man;
    if (exp == 0 && man != 0)
    {
        // Subnormal: regular mantissa shift
        // Subnormal handling: value = (-1)^sign * (man / 2^{man_bits}) * 2^{1-exp_bias}
        float frac = (float)man / (float)(1 << man_bits);
        float val = ldexpf(frac, 1 - exp_bias); // scale by 2^{1-bias}
        return sign ? -val : val;
    }
    else
    {
        // Normal: regular shift
        converted_man = man << (FP__FP32_MAN_WIDTH - man_bits);
    }

    // Build fp32
    fp__fp32_t result;
    result.f.sign = sign;
    result.f.exp = converted_exp;
    result.f.man = converted_man;
    return result.fp;
}

// Generic conversion function from fp32 to other floating-point types
// Handles all floating-point types by adjusting exponent and mantissa
uint32_t fp__convert_from_fp32(float value, int exp_bits, int man_bits)
{
    fp__fp32_t in;
    in.fp = value;

    uint8_t sign = in.f.sign;
    uint8_t fp32_exp = in.f.exp;
    uint32_t fp32_man = in.f.man;

    // Handle zero and denormalized numbers
    if (fp32_exp == 0 && fp32_man == 0)
    {
        return 0;
    }

    // Calculate bias for target format: 2^(exp_bits-1) - 1
    int target_exp_bias = (1 << (exp_bits - 1)) - 1;
    int fp32_exp_bias = FP__FP32_BIAS;

    // Convert exponent
    int32_t exp = (int32_t)fp32_exp - fp32_exp_bias + target_exp_bias;

    // Calculate max exponent value (all bits set)
    int max_exp = (1 << exp_bits) - 1;

    // Handle overflow
    if (exp >= max_exp)
    {
        exp = max_exp;
        // Return infinity or max value (all mantissa bits set)
        return (sign << (exp_bits + man_bits)) | (exp << man_bits) | ((1 << man_bits) - 1);
    }

    // Handle underflow - when exp goes negative or equals 0, clamp to 0 (subnormal)
    // For true subnormals (exp <= 0), adjust mantissa extraction
    if (exp <= 0)
    {
        // Subnormal: clamp exponent to 0, extract mantissa normally
        uint32_t man = fp32_man | (1 << FP__FP32_MAN_WIDTH);
        man >>= (FP__FP32_MAN_WIDTH - man_bits);
        man = (man >> (1 - exp)) & ((1 << man_bits) - 1);
        uint32_t result = (sign << (exp_bits + man_bits)) | (0 << man_bits) | man;
        return result;
    }

    // Extract top bits of mantissa
    // fp32 mantissa is 23 bits, we need man_bits bits
    uint32_t man = (fp32_man >> (FP__FP32_MAN_WIDTH - man_bits)) & ((1 << man_bits) - 1);

    // Build result: sign | exponent | mantissa
    uint32_t result = (sign << (exp_bits + man_bits)) | (exp << man_bits) | man;

    return result;
}

// Conversion functions using the generic converter

float fp__fp8_e4m3_to_fp32(fp__fp8_e4m3_t a)
{
    return fp__convert_to_fp32(a.f.sign, a.f.exp, a.f.man, 4, 3);
}

float fp__fp8_e5m2_to_fp32(fp__fp8_e5m2_t a)
{
    return fp__convert_to_fp32(a.f.sign, a.f.exp, a.f.man, 5, 2);
}

float fp__tf32_to_fp32(fp__tf32_t a)
{
    return fp__convert_to_fp32(a.f.sign, a.f.exp, a.f.man, 8, 10);
}

// Conversion functions from fp32 to other formats using generic converter

fp__fp8_e4m3_t fp__fp32_to_fp8_e4m3(float a)
{
    uint32_t raw = fp__convert_from_fp32(a, 4, 3);
    fp__fp8_e4m3_t result;
    result.w = (uint8_t)raw;
    return result;
}

fp__fp8_e5m2_t fp__fp32_to_fp8_e5m2(float a)
{
    uint32_t raw = fp__convert_from_fp32(a, 5, 2);
    fp__fp8_e5m2_t result;
    result.w = (uint8_t)raw;
    return result;
}

fp__tf32_t fp__fp32_to_tf32(float a)
{
    uint32_t raw = fp__convert_from_fp32(a, 8, 10);
    fp__tf32_t result;
    result.w = raw;
    return result;
}

// Multiplication functions - all use the same pattern
float fp__fp8_e4m3_mult(fp__fp8_e4m3_t a, fp__fp8_e4m3_t b)
{
    float a32 = fp__fp8_e4m3_to_fp32(a);
    float b32 = fp__fp8_e4m3_to_fp32(b);
    float res = a32 * b32;
    return res;
}

float fp__fp8_e5m2_mult(fp__fp8_e5m2_t a, fp__fp8_e5m2_t b)
{
    return fp__fp8_e5m2_to_fp32(a) * fp__fp8_e5m2_to_fp32(b);
}

float fp__tf32_mult(fp__tf32_t a, fp__tf32_t b)
{
    return fp__tf32_to_fp32(a) * fp__tf32_to_fp32(b);
}