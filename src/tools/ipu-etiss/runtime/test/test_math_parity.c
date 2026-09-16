/* test_math_parity.c — standalone bit-exact parity harness.
 *
 * Reads the golden file produced by gen_golden.py (which runs the *Python*
 * reference implementations), recomputes every case with the C port, and
 * compares the raw IEEE-754 bit patterns.  Any difference at all — including a
 * single ULP or a NaN payload — is a failure.
 *
 * Build:
 *   gcc -O2 -std=c99 -I<runtime dir> test_math_parity.c ipu_math.c \
 *       ipu_activations.c -lm -o test_math_parity
 * Run:
 *   ./test_math_parity <golden-file>
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ipu_math.h"
#include "ipu_activations.h"

#define MAX_SHOWN_MISMATCHES 10

static double bits_to_double(uint64_t u)
{
    double d;
    memcpy(&d, &u, sizeof d);
    return d;
}

static uint64_t double_to_bits(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof u);
    return u;
}

struct stats {
    unsigned long checked;
    unsigned long failed;
};

static struct stats g_group[8];
static const char *const kGroupName[8] = {
    "FP8DEC", "FP8ENC", "ONEBYTE", "MULT", "ADD", "SUB", "WRAP32", "ACT",
};
enum { G_DEC, G_ENC, G_ONE, G_MULT, G_ADD, G_SUB, G_WRAP, G_ACT };

static unsigned long g_total_fail;

static void report(int group, unsigned long line, const char *what,
                   const char *inputs, uint64_t expected, uint64_t actual)
{
    g_group[group].failed++;
    g_total_fail++;
    if (g_total_fail <= MAX_SHOWN_MISMATCHES) {
        fprintf(stderr,
                "MISMATCH #%lu (line %lu) %s: %s\n"
                "    expected 0x%016" PRIx64 " (%.17g)\n"
                "    actual   0x%016" PRIx64 " (%.17g)\n",
                g_total_fail, line, what, inputs, expected,
                bits_to_double(expected), actual, bits_to_double(actual));
    }
}

static void report_int(int group, unsigned long line, const char *what,
                       const char *inputs, long expected, long actual)
{
    g_group[group].failed++;
    g_total_fail++;
    if (g_total_fail <= MAX_SHOWN_MISMATCHES) {
        fprintf(stderr,
                "MISMATCH #%lu (line %lu) %s: %s\n"
                "    expected %ld\n"
                "    actual   %ld\n",
                g_total_fail, line, what, inputs, expected, actual);
    }
}

int main(int argc, char **argv)
{
    FILE *fh;
    char line[512];
    char desc[256];
    unsigned long lineno = 0;
    unsigned long declared = 0;
    unsigned long total = 0;
    int i;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <golden-file>\n", argv[0]);
        return 2;
    }
    fh = fopen(argv[1], "r");
    if (!fh) {
        perror(argv[1]);
        return 2;
    }

    while (fgets(line, sizeof line, fh)) {
        char tag[16];
        lineno++;
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') {
            continue;
        }
        if (sscanf(line, "%15s", tag) != 1) {
            continue;
        }

        if (strcmp(tag, "COUNT") == 0) {
            sscanf(line, "%*s %lu", &declared);
            continue;
        }

        if (strcmp(tag, "FP8DEC") == 0) {
            unsigned byte_val, exp_bits;
            uint64_t want;
            uint64_t got;
            if (sscanf(line, "%*s %u %u %" SCNx64, &byte_val, &exp_bits, &want) != 3) {
                fprintf(stderr, "bad FP8DEC line %lu: %s", lineno, line);
                return 2;
            }
            got = double_to_bits(ipu_fp8_to_double((uint8_t)byte_val, (int)exp_bits));
            g_group[G_DEC].checked++;
            total++;
            if (got != want) {
                snprintf(desc, sizeof desc, "byte=0x%02x exp_bits=%u", byte_val, exp_bits);
                report(G_DEC, lineno, "ipu_fp8_to_double", desc, want, got);
            }
        } else if (strcmp(tag, "FP8ENC") == 0) {
            uint64_t in_bits;
            unsigned exp_bits, want;
            unsigned got;
            if (sscanf(line, "%*s %" SCNx64 " %u %u", &in_bits, &exp_bits, &want) != 3) {
                fprintf(stderr, "bad FP8ENC line %lu: %s", lineno, line);
                return 2;
            }
            got = ipu_double_to_fp8(bits_to_double(in_bits), (int)exp_bits);
            g_group[G_ENC].checked++;
            total++;
            if (got != want) {
                snprintf(desc, sizeof desc, "val=%.17g (0x%016" PRIx64 ") exp_bits=%u",
                         bits_to_double(in_bits), in_bits, exp_bits);
                report_int(G_ENC, lineno, "ipu_double_to_fp8", desc, (long)want, (long)got);
            }
        } else if (strcmp(tag, "ONEBYTE") == 0) {
            unsigned dtype, want, got;
            if (sscanf(line, "%*s %u %u", &dtype, &want) != 2) {
                fprintf(stderr, "bad ONEBYTE line %lu: %s", lineno, line);
                return 2;
            }
            got = ipu_dtype_one_byte((int)dtype);
            g_group[G_ONE].checked++;
            total++;
            if (got != want) {
                snprintf(desc, sizeof desc, "dtype=%u", dtype);
                report_int(G_ONE, lineno, "ipu_dtype_one_byte", desc, (long)want, (long)got);
            }
        } else if (strcmp(tag, "MULT") == 0) {
            unsigned a, b, dtype;
            uint64_t want, got;
            if (sscanf(line, "%*s %u %u %u %" SCNx64, &a, &b, &dtype, &want) != 4) {
                fprintf(stderr, "bad MULT line %lu: %s", lineno, line);
                return 2;
            }
            got = double_to_bits(ipu_mult((uint8_t)a, (uint8_t)b, (int)dtype));
            g_group[G_MULT].checked++;
            total++;
            if (got != want) {
                snprintf(desc, sizeof desc, "a=0x%02x b=0x%02x dtype=%u", a, b, dtype);
                report(G_MULT, lineno, "ipu_mult", desc, want, got);
            }
        } else if (strcmp(tag, "ADD") == 0 || strcmp(tag, "SUB") == 0) {
            int is_add = (tag[0] == 'A');
            int grp = is_add ? G_ADD : G_SUB;
            uint64_t ab, bb, want, got;
            unsigned dtype;
            if (sscanf(line, "%*s %" SCNx64 " %" SCNx64 " %u %" SCNx64,
                       &ab, &bb, &dtype, &want) != 4) {
                fprintf(stderr, "bad ADD/SUB line %lu: %s", lineno, line);
                return 2;
            }
            got = double_to_bits(is_add
                                     ? ipu_add(bits_to_double(ab), bits_to_double(bb), (int)dtype)
                                     : ipu_sub(bits_to_double(ab), bits_to_double(bb), (int)dtype));
            g_group[grp].checked++;
            total++;
            if (got != want) {
                snprintf(desc, sizeof desc, "a=%.17g b=%.17g dtype=%u",
                         bits_to_double(ab), bits_to_double(bb), dtype);
                report(grp, lineno, is_add ? "ipu_add" : "ipu_sub", desc, want, got);
            }
        } else if (strcmp(tag, "WRAP32") == 0) {
            uint64_t in_bits;
            long want;
            long got;
            if (sscanf(line, "%*s %" SCNx64 " %ld", &in_bits, &want) != 2) {
                fprintf(stderr, "bad WRAP32 line %lu: %s", lineno, line);
                return 2;
            }
            got = (long)ipu_wrap_int32(bits_to_double(in_bits));
            g_group[G_WRAP].checked++;
            total++;
            if (got != want) {
                snprintf(desc, sizeof desc, "v=%.17g", bits_to_double(in_bits));
                report_int(G_WRAP, lineno, "ipu_wrap_int32", desc, want, got);
            }
        } else if (strcmp(tag, "ACT") == 0) {
            long fn;
            uint64_t xb, ab, want, got;
            if (sscanf(line, "%*s %ld %" SCNx64 " %" SCNx64 " %" SCNx64,
                       &fn, &xb, &ab, &want) != 4) {
                fprintf(stderr, "bad ACT line %lu: %s", lineno, line);
                return 2;
            }
            got = double_to_bits(ipu_apply_activation((int)fn, bits_to_double(xb),
                                                      bits_to_double(ab)));
            g_group[G_ACT].checked++;
            total++;
            if (got != want) {
                const char *nm = ipu_activation_name((int)fn);
                snprintf(desc, sizeof desc, "fn=%ld (%s) x=%.17g alpha=%.17g",
                         fn, nm ? nm : "<oob>", bits_to_double(xb), bits_to_double(ab));
                report(G_ACT, lineno, "ipu_apply_activation", desc, want, got);
            }
        } else {
            fprintf(stderr, "unknown record on line %lu: %s", lineno, line);
            return 2;
        }
    }
    fclose(fh);

    if (declared != 0 && declared != total) {
        fprintf(stderr, "ERROR: golden file declared %lu cases but %lu were read\n",
                declared, total);
        return 2;
    }
    if (total == 0) {
        fprintf(stderr, "ERROR: golden file contained no cases\n");
        return 2;
    }

    printf("\n");
    printf("%-10s %12s %12s\n", "group", "checked", "mismatches");
    printf("---------- ------------ ------------\n");
    for (i = 0; i < 8; i++) {
        if (g_group[i].checked == 0) {
            continue;
        }
        printf("%-10s %12lu %12lu\n", kGroupName[i], g_group[i].checked, g_group[i].failed);
    }
    printf("---------- ------------ ------------\n");
    printf("%-10s %12lu %12lu\n", "TOTAL", total, g_total_fail);
    printf("\n");

    if (g_total_fail != 0) {
        if (g_total_fail > MAX_SHOWN_MISMATCHES) {
            fprintf(stderr, "(%lu further mismatches not shown)\n",
                    g_total_fail - MAX_SHOWN_MISMATCHES);
        }
        printf("FAIL: %lu of %lu cases differ from the Python reference\n", g_total_fail, total);
        return 1;
    }
    printf("PASS: all %lu cases match the Python reference bit-exactly\n", total);
    return 0;
}
