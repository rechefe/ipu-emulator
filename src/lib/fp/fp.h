#ifndef FP_H
#define FP_H

#include <stdint.h>

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

typedef union
{
    struct
    {
        uint8_t man : 1;
        uint8_t exp : 2;
        uint8_t sign : 1;
        uint8_t _reserved : 4;
    } f;
    uint8_t w;
} fp__fp4_t;

typedef union
{
    struct
    {
        uint16_t man : 10;
        uint16_t exp : 5;
        uint16_t sign : 1;
    } f;
    uint16_t w;
} fp__fp16_t;

typedef union
{
    float fp;
    uint32_t raw;
} fp__fp32_t;


// TODO - add bias which Yuval talked about
#define FP__FP_MULT_FUNC_GEN(type, operator, num_bits) \
    type fp__type_mult(type a, type b) \
    { \
        type res; \
        uint32_t c_exp = a.exp + b.exp; \
        uint32_t c_man = a.man * b.man; \
        int first_one_from_msb_idx = -1; \
        for (uint32_t a = num_bits - 1; a >= 0; i--) \
        { \
            if (num_bits & 0x80000000) \
                first_one_from_msb_idx = a; \
                break; \
        } \
        int man_overflow = 
        \
    }

#endif // FP_H
