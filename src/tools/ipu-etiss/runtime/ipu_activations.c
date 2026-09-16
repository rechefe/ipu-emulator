/* ipu_activations.c — C99 port of ipu_common/activations.py (bit-exact).
 *
 * Every helper and branch mirrors the Python module, including the two-branch
 * numerically-stable sigmoid, the erf-based (exact, not tanh-approximated)
 * GELU, and the three-way softplus.  Python's math.exp/tanh/erf/log1p/log/sqrt
 * all forward to the platform libm on Linux, so calling the same libm entry
 * points in double precision reproduces Python bit for bit.  Single-precision
 * variants (expf, tanhf, ...) must never be used here.
 */

#include "ipu_activations.h"

#include <math.h>
#include <stdint.h>

/* Assembly / encoding order (id = index); must match the ids in the header. */
static const char *const kActivationNames[IPU_ACTIVATION_COUNT] = {
    "identity",
    "relu",
    "relu6",
    "sigmoid",
    "tanh",
    "gelu",
    "softplus",
    "elu",
    "exp2",
    "reciprocal",
    "rsqrt",
    "silu",
};

static double act_sigmoid(double x)
{
    double z;
    if (x >= 0.0) {
        z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    z = exp(x);
    return z / (1.0 + z);
}

static double act_norm_cdf(double x)
{
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

static double act_softplus(double x)
{
    /* log(1+exp(x)); stable for large |x| */
    if (x > 20.0) {
        return x;
    }
    if (x < -20.0) {
        return exp(x);
    }
    return log1p(exp(x));
}

const char *ipu_activation_name(int fn_id)
{
    uint32_t k = (uint32_t)fn_id;
    if (k >= (uint32_t)IPU_ACTIVATION_COUNT) {
        return 0;
    }
    return kActivationNames[k];
}

double ipu_apply_activation(int fn_id, double x, double elu_alpha)
{
    /* k = int(fn_id) & 0xFFFFFFFF; negative ids become huge and fall through
       to identity, exactly as in Python. */
    uint32_t k = (uint32_t)fn_id;
    double ea = elu_alpha;

    if (k >= (uint32_t)IPU_ACTIVATION_COUNT) {
        return x;
    }

    switch (k) {
    case IPU_ACTIVATION_IDENTITY:
        return x;
    case IPU_ACTIVATION_RELU:
        /* x if x > 0.0 else 0.0  (NaN -> 0.0) */
        return (x > 0.0) ? x : 0.0;
    case IPU_ACTIVATION_RELU6: {
        /* min(max(x, 0.0), 6.0) with Python's min/max NaN semantics:
           max(x, 0.0) keeps x unless 0.0 > x; min(t, 6.0) keeps t unless
           6.0 < t.  NaN therefore propagates. */
        double t = (0.0 > x) ? 0.0 : x;
        return (6.0 < t) ? 6.0 : t;
    }
    case IPU_ACTIVATION_SIGMOID:
        return act_sigmoid(x);
    case IPU_ACTIVATION_TANH:
        return tanh(x);
    case IPU_ACTIVATION_GELU:
        return x * act_norm_cdf(x);
    case IPU_ACTIVATION_SOFTPLUS:
        return act_softplus(x);
    case IPU_ACTIVATION_ELU:
        /* x if x >= 0.0 else ea * (exp(x) - 1.0)  (NaN takes the else branch) */
        return (x >= 0.0) ? x : ea * (exp(x) - 1.0);
    case IPU_ACTIVATION_EXP2:
        return exp(x * log(2.0));
    case IPU_ACTIVATION_RECIPROCAL:
        return (x != 0.0) ? (1.0 / x) : 0.0;
    case IPU_ACTIVATION_RSQRT:
        return (x > 0.0) ? (1.0 / sqrt(x)) : 0.0;
    case IPU_ACTIVATION_SILU:
        return x * act_sigmoid(x);
    default:
        return x;
    }
}
