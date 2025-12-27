#include "fp/fp.h"
#include <float.h>

// Generic conversion function that handles all floating-point types
float fp__convert_to_fp32(uint8_t sign, uint32_t exp, uint32_t man, 
                          int exp_bits, int man_bits)
{
    if (exp == 0 && man == 0) return 0.0f; // Zero
    
    // Calculate bias: 2^(exp_bits-1) - 1
    int exp_bias = (1 << (exp_bits - 1)) - 1;
    int fp32_bias = 127;
    
    // Convert exponent
    int converted_exp = (int)exp - exp_bias + fp32_bias;
    
    // Convert mantissa - shift to occupy 23 bits
    uint32_t converted_man = man << (23 - man_bits);
    
    // Build fp32
    fp__fp32_t result;
    result.raw = ((uint32_t)sign << 31) | ((uint32_t)converted_exp << 23) | converted_man;
    return result.fp;
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

float fp__fp4_to_fp32(fp__fp4_t a)
{
    return fp__convert_to_fp32(a.f.sign, a.f.exp, a.f.man, 2, 1);
}

float fp__fp16_to_fp32(fp__fp16_t a)
{
    return fp__convert_to_fp32(a.f.sign, a.f.exp, a.f.man, 5, 10);
}

// Multiplication functions - all use the same pattern

float fp__fp8_e4m3_mult(fp__fp8_e4m3_t a, fp__fp8_e4m3_t b)
{
    return fp__fp8_e4m3_to_fp32(a) * fp__fp8_e4m3_to_fp32(b);
}

float fp__fp8_e5m2_mult(fp__fp8_e5m2_t a, fp__fp8_e5m2_t b)
{
    return fp__fp8_e5m2_to_fp32(a) * fp__fp8_e5m2_to_fp32(b);
}

float fp__fp4_mult(fp__fp4_t a, fp__fp4_t b)
{
    return fp__fp4_to_fp32(a) * fp__fp4_to_fp32(b);
}

float fp__fp16_mult(fp__fp16_t a, fp__fp16_t b)
{
    return fp__fp16_to_fp32(a) * fp__fp16_to_fp32(b);
}

// Addition functions - all use the same pattern

float fp__fp8_e4m3_add(fp__fp8_e4m3_t a, fp__fp8_e4m3_t b)
{
    return fp__fp8_e4m3_to_fp32(a) + fp__fp8_e4m3_to_fp32(b);
}

float fp__fp8_e5m2_add(fp__fp8_e5m2_t a, fp__fp8_e5m2_t b)
{
    return fp__fp8_e5m2_to_fp32(a) + fp__fp8_e5m2_to_fp32(b);
}

float fp__fp4_add(fp__fp4_t a, fp__fp4_t b)
{
    return fp__fp4_to_fp32(a) + fp__fp4_to_fp32(b);
}

float fp__fp16_add(fp__fp16_t a, fp__fp16_t b)
{
    return fp__fp16_to_fp32(a) + fp__fp16_to_fp32(b);
}
