/* ipu_math.h — C99 port of ipu_emu/ipu_math.py.
 *
 * Bit-exact parity with the Python reference implementation is the contract of
 * this file: every routine below mirrors, statement for statement, the function
 * of the same name in
 *   src/tools/ipu-emu-py/src/ipu_emu/ipu_math.py
 *
 * All FP8 formats are the generic e(x)m(8-x) encoding:
 *   1 sign bit | x exponent bits | (7-x) mantissa bits, bias = 2^(x-1) - 1.
 * An all-ones exponent encodes NaN; a zero exponent encodes subnormals.
 * dtype 0 is the integer (INT8) mode and has no FP8 encoding.
 */
#ifndef IPU_MATH_H
#define IPU_MATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* dtype: 0 = INT8 (integer mode); 1..7 = FP8 with that many exponent bits */
#define IPU_DTYPE_INT8 0

double  ipu_fp8_to_double(uint8_t byte_val, int exp_bits);   /* port of _fp8_to_float32_scalar */
uint8_t ipu_double_to_fp8(double val, int exp_bits);          /* port of _float32_to_fp8_scalar */
uint8_t ipu_dtype_one_byte(int dtype);                        /* port of dtype_one_byte */
/* ipu_mult: INT8 -> integer product as double; FP8 -> product of decoded values */
double  ipu_mult(uint8_t a, uint8_t b, int dtype);            /* port of ipu_mult */
double  ipu_add(double a, double b, int dtype);               /* port of ipu_add */
double  ipu_sub(double a, double b, int dtype);               /* port of ipu_sub */
int32_t ipu_int8_to_signed(uint8_t v);
int32_t ipu_wrap_int32(double v); /* two's-complement wrap of an integral double to int32 */
/* Python's round(): half-to-even, unlike C's round() which is half-away-from-zero. */
double  ipu_py_round(double x);

#ifdef __cplusplus
}
#endif

#endif /* IPU_MATH_H */
