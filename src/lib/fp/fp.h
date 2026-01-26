#ifndef FP_H
#define FP_H

#include <stdint.h>
#include <stddef.h>

typedef union
{
    struct
    {
        uint8_t man : 3;
        uint8_t exp : 4;
        uint8_t sign : 1;
    } f;
    uint8_t w;
} fp__fp8_e4m3_t;

typedef union
{
    struct
    {
        uint8_t man : 2;
        uint8_t exp : 5;
        uint8_t sign : 1;
    } f;
    uint8_t w;
} fp__fp8_e5m2_t;

#define FP__TF32_EXP_WIDTH 8
#define FP__TF32_MAN_WIDTH 10
#define FP__TF32_WIDTH (1 + FP__TF32_EXP_WIDTH + FP__TF32_MAN_WIDTH)
typedef union
{
    struct
    {
        uint32_t man : FP__TF32_MAN_WIDTH;
        uint32_t exp : FP__TF32_EXP_WIDTH;
        uint32_t sign : 1;
    } f;
    uint32_t w;
} fp__tf32_t;

#define FP__FP32_EXP_WIDTH 8
#define FP__FP32_MAN_WIDTH 23
#define FP__FP32_BIAS ((1 << (FP__FP32_EXP_WIDTH - 1)) - 1)
typedef union
{
    struct
    {
        uint32_t man : FP__FP32_MAN_WIDTH;
        uint32_t exp : FP__FP32_EXP_WIDTH;
        uint32_t sign : 1;
    } f;
    float fp;
    uint32_t raw;
} fp__fp32_t;

// Generic conversion to fp32
float fp__convert_to_fp32(uint8_t sign, uint32_t exp, uint32_t man,
                          int exp_bits, int man_bits);

// Conversion functions to fp32
float fp__fp8_e4m3_to_fp32(fp__fp8_e4m3_t a);
float fp__fp8_e5m2_to_fp32(fp__fp8_e5m2_t a);
float fp__tf32_to_fp32(fp__tf32_t a);

// Conversion functions from fp32
fp__fp8_e4m3_t fp__fp32_to_fp8_e4m3(float a);
fp__fp8_e5m2_t fp__fp32_to_fp8_e5m2(float a);
fp__tf32_t fp__fp32_to_tf32(float a);

// Multiplication functions - return fp32
float fp__fp8_e4m3_mult(fp__fp8_e4m3_t a, fp__fp8_e4m3_t b);
float fp__fp8_e5m2_mult(fp__fp8_e5m2_t a, fp__fp8_e5m2_t b);
float fp__tf32_mult(fp__tf32_t a, fp__tf32_t b);

#endif // FP_H
