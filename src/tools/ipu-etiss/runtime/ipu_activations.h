/* ipu_activations.h — C99 port of ipu_common/activations.py.
 *
 * fn_id values are the ACTIVATION_* ids 0..11 from the Python module, in the
 * same assembly/encoding order.  Unknown ids fall through to identity, exactly
 * as apply_activation() does.
 */
#ifndef IPU_ACTIVATIONS_H
#define IPU_ACTIVATIONS_H

#ifdef __cplusplus
extern "C" {
#endif

#define IPU_ACTIVATION_COUNT 12

/* Ids must match ACTIVATION_* in activations.py. */
#define IPU_ACTIVATION_IDENTITY   0
#define IPU_ACTIVATION_RELU       1
#define IPU_ACTIVATION_RELU6      2
#define IPU_ACTIVATION_SIGMOID    3
#define IPU_ACTIVATION_TANH       4
#define IPU_ACTIVATION_GELU       5
#define IPU_ACTIVATION_SOFTPLUS   6
#define IPU_ACTIVATION_ELU        7
#define IPU_ACTIVATION_EXP2       8
#define IPU_ACTIVATION_RECIPROCAL 9
#define IPU_ACTIVATION_RSQRT      10
#define IPU_ACTIVATION_SILU       11

/* fn_id matches ACTIVATION_* ids 0..11 in activations.py */
double ipu_apply_activation(int fn_id, double x, double elu_alpha);
const char *ipu_activation_name(int fn_id);

#ifdef __cplusplus
}
#endif

#endif /* IPU_ACTIVATIONS_H */
