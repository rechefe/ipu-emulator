#ifndef FP_H
#define FP_H

#include <stdint.h>
#include "xmem/xmem.h"

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

// Generic conversion to fp32
float fp__convert_to_fp32(uint8_t sign, uint32_t exp, uint32_t man,
                          int exp_bits, int man_bits);

// Conversion functions to fp32
float fp__fp8_e4m3_to_fp32(fp__fp8_e4m3_t a);
float fp__fp8_e5m2_to_fp32(fp__fp8_e5m2_t a);
float fp__fp4_to_fp32(fp__fp4_t a);
float fp__fp16_to_fp32(fp__fp16_t a);

// Multiplication functions - return fp32
float fp__fp8_e4m3_mult(fp__fp8_e4m3_t a, fp__fp8_e4m3_t b);
float fp__fp8_e5m2_mult(fp__fp8_e5m2_t a, fp__fp8_e5m2_t b);
float fp__fp4_mult(fp__fp4_t a, fp__fp4_t b);
float fp__fp16_mult(fp__fp16_t a, fp__fp16_t b);

// Addition functions - return fp32
float fp__fp8_e4m3_add(fp__fp8_e4m3_t a, fp__fp8_e4m3_t b);
float fp__fp8_e5m2_add(fp__fp8_e5m2_t a, fp__fp8_e5m2_t b);
float fp__fp4_add(fp__fp4_t a, fp__fp4_t b);
float fp__fp16_add(fp__fp16_t a, fp__fp16_t b);

// Conversion and XMEM loading functions
/**
 * @brief Load fp32 array from file and convert to specified format, then load to XMEM
 *
 * @param xmem External memory object
 * @param file_path Path to binary file containing fp32 values
 * @param format Target floating-point format (0=fp8_e4m3, 1=fp8_e5m2, 2=fp16, 3=fp4)
 * @param base_address Starting address in XMEM
 * @param chunk_size Size of each chunk to process
 * @param num_chunks Number of chunks to process (0 = entire file)
 * @return Number of converted values loaded to XMEM, or -1 on error
 */
int fp__load_fp32_file_to_xmem(
    xmem__obj_t *xmem,
    const char *file_path,
    int format,
    uint32_t base_address,
    size_t chunk_size,
    size_t num_chunks);

#endif // FP_H
