#include "fp/fp.h"  
#include "xmem/xmem.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// Generic conversion function that handles all floating-point types
float fp__convert_to_fp32(uint8_t sign, uint32_t exp, uint32_t man,
                          int exp_bits, int man_bits)
{
    if (exp == 0 && man == 0)
        return 0.0f; // Zero

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

// Generic conversion function from fp32 to other floating-point types
// Handles all floating-point types by adjusting exponent and mantissa
uint32_t fp__convert_from_fp32(float value, int exp_bits, int man_bits)
{
    fp__fp32_t in;
    in.fp = value;

    uint8_t sign = (in.raw >> 31) & 0x1;
    uint8_t fp32_exp = (in.raw >> 23) & 0xFF;
    uint32_t fp32_man = in.raw & 0x7FFFFF;

    // Handle zero and denormalized numbers
    if (fp32_exp == 0 && fp32_man == 0)
    {
        return 0;
    }

    // Calculate bias for target format: 2^(exp_bits-1) - 1
    int target_exp_bias = (1 << (exp_bits - 1)) - 1;
    int fp32_exp_bias = 127;

    // Convert exponent
    int32_t exp = (int32_t)fp32_exp - fp32_exp_bias + target_exp_bias;

    // Calculate max exponent value (all bits set)
    int max_exp = (1 << exp_bits) - 1;

    // Handle underflow/overflow
    if (exp <= 0)
    {
        exp = 0;
        return 0; // Underflow to zero
    }
    else if (exp >= max_exp)
    {
        exp = max_exp;
        // Return infinity or max value (all mantissa bits set)
        return (sign << (exp_bits + man_bits)) | (exp << man_bits) | ((1 << man_bits) - 1);
    }

    // Extract top bits of mantissa
    // fp32 mantissa is 23 bits, we need man_bits bits
    uint32_t man = (fp32_man >> (23 - man_bits)) & ((1 << man_bits) - 1);

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

fp__fp4_t fp__fp32_to_fp4(float a)
{
    uint32_t raw = fp__convert_from_fp32(a, 2, 1);
    fp__fp4_t result;
    result.w = (uint8_t)raw;
    return result;
}

fp__fp16_t fp__fp32_to_fp16(float a)
{
    uint32_t raw = fp__convert_from_fp32(a, 5, 10);
    fp__fp16_t result;
    result.w = (uint16_t)raw;
    return result;
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
    float res = fp__fp8_e4m3_to_fp32(a) * fp__fp8_e4m3_to_fp32(b);
    return res;
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

// Load fp32 file and convert to target format, then load to XMEM
int fp__load_fp32_file_to_xmem(
    xmem__obj_t *xmem,
    const char *file_path,
    int format,
    uint32_t base_address,
    size_t chunk_size,
    size_t num_chunks)
{
    if (!xmem || !file_path)
    {
        printf("Error: NULL pointer passed to fp__load_fp32_file_to_xmem\n");
        return -1;
    }

    // Open file
    FILE *file = fopen(file_path, "rb");
    if (!file)
    {
        printf("Error: Cannot open file '%s'\n", file_path);
        return -1;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Calculate number of fp32 values
    int num_fp32_values = file_size / sizeof(float);
    if (file_size % sizeof(float) != 0)
    {
        printf("Warning: File size (%ld) is not a multiple of 4 bytes\n", file_size);
    }

    // Determine output element size based on format
    int output_element_size;
    switch (format)
    {
    case 0: // fp8_e4m3
    case 1: // fp8_e5m2
    case 3: // fp4
        output_element_size = 1;
        break;
    case 2: // fp16
        output_element_size = 2;
        break;
    default:
        printf("Error: Unknown format %d\n", format);
        fclose(file);
        return -1;
    }

    // Determine how many values to process
    size_t values_to_process = num_fp32_values;
    if (num_chunks > 0)
    {
        values_to_process = (chunk_size / sizeof(float)) * num_chunks;
        if (values_to_process > num_fp32_values)
        {
            values_to_process = num_fp32_values;
        }
    }

    // Allocate temporary buffers
    float *fp32_buffer = (float *)malloc(values_to_process * sizeof(float));
    uint8_t *converted_buffer = (uint8_t *)malloc(values_to_process * output_element_size);

    if (!fp32_buffer || !converted_buffer)
    {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        free(fp32_buffer);
        free(converted_buffer);
        return -1;
    }

    // Read fp32 values from file
    size_t read_count = fread(fp32_buffer, sizeof(float), values_to_process, file);
    if (read_count != values_to_process)
    {
        printf("Warning: Read %zu values, expected %zu\n", read_count, values_to_process);
    }
    fclose(file);

    // Convert fp32 values to target format
    int converted_count = 0;
    for (size_t i = 0; i < read_count; i++)
    {
        switch (format)
        {
        case 0: // fp8_e4m3
        {
            fp__fp8_e4m3_t val = fp__fp32_to_fp8_e4m3(fp32_buffer[i]);
            converted_buffer[i] = val.w;
            break;
        }
        case 1: // fp8_e5m2
        {
            fp__fp8_e5m2_t val = fp__fp32_to_fp8_e5m2(fp32_buffer[i]);
            converted_buffer[i] = val.w;
            break;
        }
        case 2: // fp16
        {
            fp__fp16_t val = fp__fp32_to_fp16(fp32_buffer[i]);
            uint16_t *ptr = (uint16_t *)converted_buffer;
            ptr[i] = val.w;
            break;
        }
        case 3: // fp4
        {
            fp__fp4_t val = fp__fp32_to_fp4(fp32_buffer[i]);
            converted_buffer[i] = val.w;
            break;
        }
        }
        converted_count++;
    }

    // Load converted data to XMEM
    size_t total_bytes = converted_count * output_element_size;
    xmem__write_address(xmem, base_address, converted_buffer, total_bytes);

    // Cleanup
    free(fp32_buffer);
    free(converted_buffer);

    printf("Loaded %d values to XMEM at address 0x%x\n", converted_count, base_address);
    return converted_count;
}
