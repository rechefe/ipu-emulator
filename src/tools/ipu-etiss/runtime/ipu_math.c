/* ipu_math.c — C99 port of ipu_emu/ipu_math.py (bit-exact).
 *
 * The structure deliberately mirrors the Python module one helper at a time so
 * the two can be diffed side by side:
 *
 *   _int8_to_signed        -> ipu_int8_to_signed
 *   _fp8_max_finite        -> fp8_max_finite
 *   _fp8_decode_fields     -> fp8_decode_fields
 *   _fp8_magnitude         -> fp8_magnitude
 *   _fp8_to_float32_scalar -> ipu_fp8_to_double
 *   _fp8_encode_subnormal  -> fp8_encode_subnormal
 *   _fp8_encode_normal     -> fp8_encode_normal
 *   _float32_to_fp8_scalar -> ipu_double_to_fp8
 *   dtype_one_byte         -> ipu_dtype_one_byte
 *   ipu_mult/add/sub       -> ipu_mult/ipu_add/ipu_sub
 *
 * Everything is computed in `double`, because Python floats are IEEE-754
 * binary64.  Never narrow to `float` inside these routines.
 */

#include "ipu_math.h"

#include <math.h>

/* ------------------------------------------------------------------ */
/* Python round() — banker's rounding (round-half-to-even).            */
/*                                                                     */
/* C's round() is round-half-AWAY-from-zero and disagrees with Python  */
/* on every exact .5 tie, which shows up as off-by-one-ULP FP8         */
/* mantissas.  This is a literal transcription of CPython's            */
/* float___round___impl() for the ndigits=None case:                   */
/*                                                                     */
/*     rounded = round(x);                                             */
/*     if (fabs(x - rounded) == 0.5)                                   */
/*         rounded = 2.0 * round(x / 2.0);                             */
/*                                                                     */
/* ------------------------------------------------------------------ */
double ipu_py_round(double x)
{
    double rounded = round(x);
    if (fabs(x - rounded) == 0.5) {
        /* halfway case: round to even */
        rounded = 2.0 * round(x / 2.0);
    }
    return rounded;
}

/* 2.0 ** k for integral k — exact, matches Python's `2.0 ** k`. */
static double pow2i(int k)
{
    return ldexp(1.0, k);
}

/* ------------------------------------------------------------------ */

int32_t ipu_int8_to_signed(uint8_t v)
{
    /* return val if val < 128 else val - 256 */
    return (v < 128u) ? (int32_t)v : ((int32_t)v - 256);
}

/* Unsigned byte encoding of the largest finite FP8 value (positive). */
static int fp8_max_finite(int exp_bits, int man_bits)
{
    int max_exp_raw = (1 << exp_bits) - 1; /* all-ones = NaN */
    return ((max_exp_raw - 1) << man_bits) | ((1 << man_bits) - 1);
}

/* Split an FP8 byte into (sign, exp_raw, man_raw). */
static void fp8_decode_fields(uint8_t byte_val, int exp_bits,
                              int *sign, int *exp_raw, int *man_raw)
{
    int man_bits = 7 - exp_bits;
    *sign = (byte_val >> 7) & 1;
    *exp_raw = (byte_val >> man_bits) & ((1 << exp_bits) - 1);
    *man_raw = byte_val & ((1 << man_bits) - 1);
}

/* Unsigned magnitude of a normal or subnormal FP8 value. */
static double fp8_magnitude(int exp_raw, int man_raw, int exp_bits)
{
    int man_bits = 7 - exp_bits;
    int bias = (1 << (exp_bits - 1)) - 1;
    if (exp_raw == 0) {
        /* Subnormal: 0.man * 2^(1-bias) */
        return ((double)man_raw / (double)(1 << man_bits)) * pow2i(1 - bias);
    }
    /* Normal: 1.man * 2^(exp-bias) */
    return (1.0 + (double)man_raw / (double)(1 << man_bits)) * pow2i(exp_raw - bias);
}

double ipu_fp8_to_double(uint8_t byte_val, int exp_bits)
{
    int sign, exp_raw, man_raw;
    int max_exp;
    double value;

    fp8_decode_fields(byte_val, exp_bits, &sign, &exp_raw, &man_raw);
    max_exp = (1 << exp_bits) - 1;

    if (exp_raw == max_exp) {
        return (double)NAN;
    }

    value = fp8_magnitude(exp_raw, man_raw, exp_bits);
    return sign ? -value : value;
}

/* Encode a subnormal FP8 value from a positive float magnitude. */
static uint8_t fp8_encode_subnormal(double val, int exp_bits, int man_bits, int sign)
{
    int bias = (1 << (exp_bits - 1)) - 1;
    double max_man = (double)((1 << man_bits) - 1);
    /* val = (man_int / 2^man_bits) * 2^(1-bias) -> man_int = val * 2^(man_bits+bias-1) */
    double man_int = ipu_py_round(val * pow2i(man_bits + bias - 1));
    /* man_int = max(0, min(man_int, max_man)) — clamped in double so that
       arbitrarily large intermediates saturate rather than overflow. */
    if (man_int > max_man) {
        man_int = max_man;
    }
    if (man_int < 0.0) {
        man_int = 0.0;
    }
    return (uint8_t)((sign << 7) | (int)man_int);
}

/* Encode a normal FP8 value given the biased exponent and frexp fraction. */
static uint8_t fp8_encode_normal(double frac, int fp8_exp, int exp_bits, int man_bits, int sign)
{
    int max_exp_raw = (1 << exp_bits) - 1;
    int max_man = (1 << man_bits) - 1;
    /* For E7 (man_bits=0) the mantissa term collapses; for all other formats
       man_int lands in [0, 2^man_bits]. */
    int man_int = (int)ipu_py_round((2.0 * frac - 1.0) * (double)(1 << man_bits));
    if (man_int > max_man) {
        /* Rounding carried into the exponent; reset mantissa, recheck overflow. */
        man_int = 0;
        fp8_exp += 1;
    }
    if (fp8_exp >= max_exp_raw) {
        return (uint8_t)((sign << 7) | fp8_max_finite(exp_bits, man_bits));
    }
    return (uint8_t)((sign << 7) | (fp8_exp << man_bits) | man_int);
}

uint8_t ipu_double_to_fp8(double val, int exp_bits)
{
    int man_bits = 7 - exp_bits;
    int max_exp_raw = (1 << exp_bits) - 1; /* all-ones exponent = NaN */
    int sign;
    int exp;
    int bias;
    int fp8_exp;
    double frac;

    if (isnan(val)) {
        return (uint8_t)((max_exp_raw << man_bits) | 1); /* canonical NaN */
    }

    sign = 0;
    if (val < 0.0 || (val == 0.0 && copysign(1.0, val) < 0.0)) {
        sign = 1;
        val = fabs(val);
    }

    /* After the block above val is always non-negative.  Negative zero was
       caught by copysign, set sign=1, and fabs(-0.0)==0.0, so the check below
       correctly returns 0x80 (negative zero encoding). */
    if (val == 0.0) {
        return (uint8_t)(sign << 7);
    }

    if (isinf(val)) {
        return (uint8_t)((sign << 7) | fp8_max_finite(exp_bits, man_bits));
    }

    frac = frexp(val, &exp); /* val = frac * 2^exp, 0.5 <= frac < 1 */
    bias = (1 << (exp_bits - 1)) - 1;
    fp8_exp = (exp - 1) + bias; /* biased exponent (exp-1 is the IEEE exponent) */

    if (fp8_exp >= max_exp_raw) {
        return (uint8_t)((sign << 7) | fp8_max_finite(exp_bits, man_bits));
    }
    if (fp8_exp <= 0) {
        return fp8_encode_subnormal(val, exp_bits, man_bits, sign);
    }
    return fp8_encode_normal(frac, fp8_exp, exp_bits, man_bits, sign);
}

uint8_t ipu_dtype_one_byte(int dtype)
{
    if (dtype == IPU_DTYPE_INT8) {
        return 0x01;
    }
    return ipu_double_to_fp8(1.0, dtype);
}

double ipu_mult(uint8_t a, uint8_t b, int dtype)
{
    if (dtype == IPU_DTYPE_INT8) {
        return (double)((int32_t)ipu_int8_to_signed(a) * (int32_t)ipu_int8_to_signed(b));
    }
    /* dtype outside 1..7 raises ValueError in Python; callers must not do that. */
    return ipu_fp8_to_double(a, dtype) * ipu_fp8_to_double(b, dtype);
}

int32_t ipu_wrap_int32(double v)
{
    /* Two's-complement wrap: (v & 0xFFFFFFFF), then reinterpret as signed.
       fmod is exact for integral doubles, so this matches Python's
       arbitrary-precision `& 0xFFFFFFFF` for every integral input. */
    double m = fmod(v, 4294967296.0);
    if (m < 0.0) {
        m += 4294967296.0;
    }
    return (int32_t)(uint32_t)m;
}

double ipu_add(double a, double b, int dtype)
{
    if (dtype == IPU_DTYPE_INT8) {
        /* Both are int32 — wrap to 32 bits with two's complement */
        return (double)ipu_wrap_int32(a + b);
    }
    /* FP8 variants: accumulator is float */
    return a + b;
}

double ipu_sub(double a, double b, int dtype)
{
    if (dtype == IPU_DTYPE_INT8) {
        return (double)ipu_wrap_int32(a - b);
    }
    return a - b;
}
