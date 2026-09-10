/* IPU instruction semantics for the ETISS backend.
 *
 * A C port of ``ipu_emu/ipu.py`` (narrow mode).  Every handler here
 * corresponds one-to-one to an ``execute_*`` method there and is named after
 * it: ``execute_ldr_mult_reg`` -> ``ipu_ldr_mult_reg``.  The prototypes are
 * generated from ``INSTRUCTION_SPEC`` into ``IPUFuncs_gen.h``, so a handler
 * that is missing or misnamed fails to link.
 *
 * Not ported: wide-vector debug mode (``wide_vector_debug``), which is an
 * emulator-only analysis feature -- see the ETISS integration spec.
 *
 * Numeric parity with Python rests on ``ipu_math.c`` / ``ipu_activations.c``
 * (both verified bit-exact against the Python modules) plus two rules kept
 * throughout this file:
 *   - all arithmetic happens in ``double``, exactly as Python floats do, and
 *     is narrowed to ``float`` only when a lane is stored;
 *   - INT8-mode lanes are int32 with two's-complement wrap, matching
 *     ``ipu_math.ipu_add``/``ipu_sub``.
 */

#include "IPUFuncs.h"

#include <math.h>
#include <string.h>

#include "ipu_math.h"
#include "ipu_activations.h"

/* ------------------------------------------------------------------------ */
/* dstructure (CR) decoding -- mirrors ipu_emu.ipu_config.decode_dstructure   */
/* ------------------------------------------------------------------------ */

#define DS_VALID_BITS 8
#define DS_PART_BITS 5
#define DS_PAD_BITS 2
#define DS_VALID_MASK ((1u << DS_VALID_BITS) - 1u)
#define DS_PART_MASK ((1u << DS_PART_BITS) - 1u)
#define DS_PAD_MASK ((1u << DS_PAD_BITS) - 1u)
#define DS_PART_SHIFT DS_VALID_BITS
#define DS_PAD_SHIFT (DS_PART_SHIFT + DS_PART_BITS)

#define PAD_ZERO 0
#define PAD_POS_INF 1
#define PAD_NEG_INF 2

typedef struct
{
    etiss_uint32 valid_elements;
    etiss_uint32 partition;
    etiss_uint32 pad_mode;
} DStructure;

/* Partition is an enum in Python (P0/P2/P4/P8/P16); anything else raises. */
static int ds_partition_valid(etiss_uint32 p)
{
    return p == 0 || p == 2 || p == 4 || p == 8 || p == 16;
}

static int ipu_decode_dstructure(IPU *c, etiss_uint32 word, DStructure *out)
{
    out->valid_elements = word & DS_VALID_MASK;
    out->partition = (word >> DS_PART_SHIFT) & DS_PART_MASK;
    out->pad_mode = (word >> DS_PAD_SHIFT) & DS_PAD_MASK;
    if (!ds_partition_valid(out->partition) || out->pad_mode > PAD_NEG_INF)
    {
        ipu_raise(c, IPU_ERR_DSTRUCTURE_FIELD, word);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Error reporting and the cycle-start shadow copy                           */
/* ------------------------------------------------------------------------ */

void ipu_raise(IPU *c, etiss_uint32 code, etiss_uint64 detail)
{
    if (c->error_code == 0) /* first error wins, like the first raised exception */
    {
        c->error_code = code;
        c->error_detail = detail;
    }
}

void ipu_snapshot(IPU *c)
{
    memcpy(c->s_LR, c->LR, sizeof(c->LR));
    memcpy(c->s_CR, c->CR, sizeof(c->CR));
    memcpy(c->s_R, c->R, sizeof(c->R));
    memcpy(c->s_R_CYCLIC, c->R_CYCLIC, sizeof(c->R_CYCLIC));
    memcpy(c->s_R_ACC, c->R_ACC, sizeof(c->R_ACC));
    memcpy(c->s_MULT_RES, c->MULT_RES, sizeof(c->MULT_RES));
}

/* Every handler is a no-op once an error has been raised, so the first
 * failure aborts the rest of the word the way a Python exception would. */
#define IPU_GUARD(c)          \
    do                        \
    {                         \
        if ((c)->error_code)  \
            return;           \
    } while (0)

/* ------------------------------------------------------------------------ */
/* Lane helpers                                                              */
/* ------------------------------------------------------------------------ */

/* True when MULT_RES / R_ACC lanes hold float32 rather than int32. */
static int lanes_are_float(const IPU *c) { return c->dtype != IPU_DTYPE_INT8; }

static double lane_load(const etiss_uint8 *buf, unsigned i, int is_float)
{
    if (is_float)
    {
        float f;
        memcpy(&f, buf + (size_t)i * 4, 4);
        return (double)f;
    }
    {
        etiss_int32 v;
        memcpy(&v, buf + (size_t)i * 4, 4);
        return (double)v;
    }
}

static void lane_store(etiss_uint8 *buf, unsigned i, double value, int is_float)
{
    if (is_float)
    {
        float f = (float)value;
        memcpy(buf + (size_t)i * 4, &f, 4);
    }
    else
    {
        etiss_int32 v = ipu_wrap_int32(value);
        memcpy(buf + (size_t)i * 4, &v, 4);
    }
}

/* ------------------------------------------------------------------------ */
/* XMEM access                                                               */
/* ------------------------------------------------------------------------ */

/* Narrow mode addresses the first 16384 rows (2 MB) of the 8 MB allocation. */
#define IPU_NARROW_MAX_ROW (IPU_XMEM_SIZE / ((etiss_uint64)IPU_LANES * 4))

/* Translate an assembly XMEM row number to a byte address.
 * Mirrors Ipu._xmem_row_addr: rows are LANES elements, one byte each. */
static int xmem_row_addr(IPU *c, etiss_uint64 row, etiss_uint64 *addr)
{
    if (row >= IPU_NARROW_MAX_ROW)
    {
        ipu_raise(c, IPU_ERR_XMEM_ROW_RANGE, row);
        return -1;
    }
    *addr = row * (etiss_uint64)IPU_XMEM_WIDTH;
    return 0;
}

static int xmem_read(IPU *c, ETISS_System *sys, etiss_uint64 addr, etiss_uint8 *buf, etiss_uint32 len)
{
    if (addr + len > IPU_XMEM_SIZE)
    {
        ipu_raise(c, IPU_ERR_XMEM_ACCESS, addr);
        return -1;
    }
    if (sys->dread(sys->handle, &c->cpu, IPU_XMEM_BASE + addr, buf, len) != 0)
    {
        ipu_raise(c, IPU_ERR_XMEM_ACCESS, addr);
        return -1;
    }
    return 0;
}

static int xmem_write(IPU *c, ETISS_System *sys, etiss_uint64 addr, const etiss_uint8 *buf, etiss_uint32 len)
{
    if (addr + len > IPU_XMEM_SIZE)
    {
        ipu_raise(c, IPU_ERR_XMEM_ACCESS, addr);
        return -1;
    }
    if (sys->dwrite(sys->handle, &c->cpu, IPU_XMEM_BASE + addr, (etiss_uint8 *)buf, len) != 0)
    {
        ipu_raise(c, IPU_ERR_XMEM_ACCESS, addr);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* R_CYCLIC wrapping access (RegFile's generated get_/set_r_cyclic_at)        */
/* ------------------------------------------------------------------------ */

static void cyclic_read(const etiss_uint8 *reg, etiss_uint64 start, etiss_uint8 *out, unsigned len)
{
    unsigned sz = IPU_R_CYCLIC_SIZE;
    unsigned s = (unsigned)(start % sz);
    if (s + len <= sz)
    {
        memcpy(out, reg + s, len);
        return;
    }
    {
        unsigned first = sz - s;
        memcpy(out, reg + s, first);
        memcpy(out + first, reg, len - first);
    }
}

static void cyclic_write(etiss_uint8 *reg, etiss_uint64 start, const etiss_uint8 *in, unsigned len)
{
    unsigned sz = IPU_R_CYCLIC_SIZE;
    unsigned s = (unsigned)(start % sz);
    if (s + len <= sz)
    {
        memcpy(reg + s, in, len);
        return;
    }
    {
        unsigned first = sz - s;
        memcpy(reg + s, in, first);
        memcpy(reg, in + first, len - first);
    }
}

/* ------------------------------------------------------------------------ */
/* LRDn -- the 8 byte elements of the LR(2n+1):LR(2n) pair                    */
/* ------------------------------------------------------------------------ */

static void lrd_get(const etiss_uint32 *lr, etiss_uint32 n, etiss_uint8 *out)
{
    etiss_uint32 lo = lr[2 * n], hi = lr[2 * n + 1];
    out[0] = (etiss_uint8)(lo);
    out[1] = (etiss_uint8)(lo >> 8);
    out[2] = (etiss_uint8)(lo >> 16);
    out[3] = (etiss_uint8)(lo >> 24);
    out[4] = (etiss_uint8)(hi);
    out[5] = (etiss_uint8)(hi >> 8);
    out[6] = (etiss_uint8)(hi >> 16);
    out[7] = (etiss_uint8)(hi >> 24);
}

static void lrd_set(etiss_uint32 *lr, etiss_uint32 n, const etiss_uint8 *in)
{
    lr[2 * n] = (etiss_uint32)in[0] | ((etiss_uint32)in[1] << 8) | ((etiss_uint32)in[2] << 16) |
                ((etiss_uint32)in[3] << 24);
    lr[2 * n + 1] = (etiss_uint32)in[4] | ((etiss_uint32)in[5] << 8) | ((etiss_uint32)in[6] << 16) |
                    ((etiss_uint32)in[7] << 24);
}

/* ------------------------------------------------------------------------ */
/* Multiply masking -- Ipu._mult_mask_and_shift                              */
/* ------------------------------------------------------------------------ */

typedef unsigned __int128 mask128;

#define MASK128_ALL ((mask128)~(mask128)0 >> (128 - IPU_LANES))

/* 0 at the START of each partition group (used for left shifts). */
static mask128 partition_vector(etiss_uint32 num_partitions)
{
    mask128 result = 0;
    unsigned step, i;
    if (num_partitions == 0)
        return MASK128_ALL;
    step = IPU_LANES / num_partitions;
    for (i = 0; i < IPU_LANES; ++i)
        if (i % step != 0)
            result |= ((mask128)1) << i;
    return result;
}

/* 0 at the END of each partition group (used for right shifts). */
static mask128 inverse_partition_vector(etiss_uint32 num_partitions)
{
    mask128 result = 0;
    unsigned step, i;
    if (num_partitions == 0)
        return MASK128_ALL;
    step = IPU_LANES / num_partitions;
    for (i = 0; i < IPU_LANES; ++i)
        if (i % step != step - 1)
            result |= ((mask128)1) << i;
    return result;
}

/* The 4-byte MULT_RES fill value for a lane the mask deactivates. */
static int mult_pad_lane(IPU *c, etiss_uint32 pad_mode, etiss_uint8 out[4])
{
    if (pad_mode == PAD_ZERO)
    {
        memset(out, 0, 4);
        return 0;
    }
    if (!lanes_are_float(c))
    {
        ipu_raise(c, IPU_ERR_PAD_MODE_NEEDS_FLOAT, pad_mode);
        return -1;
    }
    {
        float v = (pad_mode == PAD_POS_INF) ? (float)INFINITY : -(float)INFINITY;
        memcpy(out, &v, 4);
    }
    return 0;
}

static void mult_mask_and_shift(IPU *c, ETISS_System *sys, etiss_uint32 mask_idx, etiss_uint32 mask_shift,
                               etiss_uint32 cr_word)
{
    DStructure ds;
    mask128 base_mask = 0, mask_int;
    etiss_int32 shift;
    unsigned slot, offset, i;
    etiss_uint8 pad[4];

    (void)sys;
    if (c->error_code)
        return;

    /* LR values are 32 bits wide; sign-extend before clamping to [-3, 3]. */
    shift = (etiss_int32)mask_shift;
    if (shift < -3)
        shift = -3;
    else if (shift > 3)
        shift = 3;

    /* R_MASK holds eight 128-bit slots; the index selects one, little-endian. */
    slot = (unsigned)(mask_idx % (IPU_LANES / 16));
    offset = slot * 16;
    for (i = 0; i < 16; ++i)
        base_mask |= ((mask128)c->R_MASK[offset + i]) << (8 * i);
    base_mask &= MASK128_ALL;

    if (ipu_decode_dstructure(c, cr_word, &ds) != 0)
        return;

    mask_int = base_mask;
    if (shift < 0)
    {
        mask128 pv = inverse_partition_vector(ds.partition);
        etiss_int32 n;
        for (n = 0; n < -shift; ++n)
            mask_int = (mask_int >> 1) & pv;
    }
    else if (shift > 0)
    {
        mask128 pv = partition_vector(ds.partition);
        etiss_int32 n;
        for (n = 0; n < shift; ++n)
            mask_int = ((mask_int << 1) & pv) & MASK128_ALL;
    }

    if (mask_int == MASK128_ALL)
        return; /* every lane active: nothing to pad (the common case) */

    if (mult_pad_lane(c, ds.pad_mode, pad) != 0)
        return;
    for (i = 0; i < IPU_LANES; ++i)
        if (!((mask_int >> i) & 1))
            memcpy(c->MULT_RES + (size_t)i * 4, pad, 4);
}

/* ======================================================================== */
/* LOAD / STORE / ACC_STORE slot                                            */
/* ======================================================================== */

void ipu_ldr_mult_reg(IPU *c, ETISS_System *sys, etiss_uint32 dest, etiss_uint32 offset, etiss_uint32 base)
{
    etiss_uint64 addr;
    IPU_GUARD(c);
    if (dest > 1)
    {
        ipu_raise(c, IPU_ERR_LDR_MULT_REG_DEST, dest);
        return;
    }
    if (xmem_row_addr(c, (etiss_uint64)offset + base, &addr) != 0)
        return;
    xmem_read(c, sys, addr, c->R + (size_t)dest * IPU_R_SIZE, IPU_R_SIZE);
}

void ipu_ldr_cyclic_mult_reg(IPU *c, ETISS_System *sys, etiss_uint32 offset, etiss_uint32 base, etiss_uint32 index)
{
    etiss_uint64 addr;
    etiss_uint8 row[IPU_R_SIZE];
    IPU_GUARD(c);
    if (xmem_row_addr(c, (etiss_uint64)offset + base, &addr) != 0)
        return;
    /* Writes replace a whole slot, so the index must land on a boundary. */
    if (index % IPU_R_SIZE != 0 || index >= IPU_R_CYCLIC_SIZE)
    {
        ipu_raise(c, IPU_ERR_CYCLIC_INDEX, index);
        return;
    }
    if (xmem_read(c, sys, addr, row, IPU_R_SIZE) != 0)
        return;
    cyclic_write(c->R_CYCLIC, index, row, IPU_R_SIZE);
}

void ipu_ldr_mult_mask_reg(IPU *c, ETISS_System *sys, etiss_uint32 offset, etiss_uint32 base)
{
    etiss_uint64 addr;
    IPU_GUARD(c);
    if (xmem_row_addr(c, (etiss_uint64)offset + base, &addr) != 0)
        return;
    xmem_read(c, sys, addr, c->R_MASK, IPU_R_MASK_SIZE);
}

void ipu_str_post_aaq_reg(IPU *c, ETISS_System *sys, etiss_uint32 offset, etiss_uint32 base)
{
    etiss_uint64 addr;
    IPU_GUARD(c);
    if (xmem_row_addr(c, (etiss_uint64)offset + base, &addr) != 0)
        return;
    xmem_write(c, sys, addr, c->POST_AAQ_REG, IPU_POST_AAQ_REG_SIZE);
}

void ipu_str_acc_reg(IPU *c, ETISS_System *sys, etiss_uint32 offset, etiss_uint32 base)
{
    etiss_uint64 addr;
    IPU_GUARD(c);
    if (xmem_row_addr(c, (etiss_uint64)offset + base, &addr) != 0)
        return;
    xmem_write(c, sys, addr, c->R_ACC, IPU_R_ACC_SIZE);
}

/* ======================================================================== */
/* LR slot                                                                  */
/* ======================================================================== */

void ipu_lr_set(IPU *c, ETISS_System *sys, etiss_uint32 reg, etiss_uint32 src)
{
    (void)sys;
    IPU_GUARD(c);
    c->LR[reg] = src;
}

void ipu_lr_add(IPU *c, ETISS_System *sys, etiss_uint32 dest, etiss_uint32 src_a, etiss_uint32 src_b)
{
    (void)sys;
    IPU_GUARD(c);
    c->LR[dest] = src_a + src_b;
}

void ipu_lr_sub(IPU *c, ETISS_System *sys, etiss_uint32 dest, etiss_uint32 src_a, etiss_uint32 src_b)
{
    (void)sys;
    IPU_GUARD(c);
    c->LR[dest] = src_a - src_b;
}

void ipu_lr_inc(IPU *c, ETISS_System *sys, etiss_uint32 dest, etiss_uint32 imm)
{
    (void)sys;
    IPU_GUARD(c);
    c->LR[dest] = c->s_LR[dest] + imm; /* read-modify-write from the snapshot */
}

void ipu_lr_dec(IPU *c, ETISS_System *sys, etiss_uint32 dest, etiss_uint32 imm)
{
    (void)sys;
    IPU_GUARD(c);
    c->LR[dest] = c->s_LR[dest] - imm;
}

/* INCR_MOD_POW2: dest <- (dest + step) mod 2^(k+1), old dest from the snapshot. */
void ipu_lr_incr_mod_pow2(IPU *c, ETISS_System *sys, etiss_uint32 dest, etiss_uint32 step, etiss_uint32 k)
{
    etiss_uint32 k_exp, mask;
    (void)sys;
    IPU_GUARD(c);
    if (k > IPU_LR_MOD_POW2_K_ENCODED_MAX)
    {
        ipu_raise(c, IPU_ERR_INCR_MOD_POW2_K, k);
        return;
    }
    k_exp = k + IPU_LR_MOD_POW2_K_MIN;
    mask = (k_exp >= 32) ? 0xFFFFFFFFu : ((1u << k_exp) - 1u);
    c->LR[dest] = (c->s_LR[dest] + step) & mask;
}

/* Broadcast-add a signed byte to LRDn's 8 byte elements, clamped to [0, 255]. */
static void addb_broadcast(IPU *c, etiss_uint32 dest, etiss_uint32 byte_val)
{
    etiss_uint8 lanes[8];
    int signed_val = (byte_val & 0xFF) >= 128 ? (int)(byte_val & 0xFF) - 256 : (int)(byte_val & 0xFF);
    unsigned i;
    lrd_get(c->s_LR, dest, lanes);
    for (i = 0; i < 8; ++i)
    {
        int v = (int)lanes[i] + signed_val;
        if (v < 0)
            v = 0;
        else if (v > 255)
            v = 255;
        lanes[i] = (etiss_uint8)v;
    }
    lrd_set(c->LR, dest, lanes);
}

void ipu_addb(IPU *c, ETISS_System *sys, etiss_uint32 dest, etiss_uint32 src_b)
{
    (void)sys;
    IPU_GUARD(c);
    addb_broadcast(c, dest, src_b & 0xFF);
}

void ipu_addbi(IPU *c, ETISS_System *sys, etiss_uint32 dest, etiss_uint32 imm)
{
    (void)sys;
    IPU_GUARD(c);
    addb_broadcast(c, dest, imm);
}

/* ======================================================================== */
/* MULT slot                                                                */
/* ======================================================================== */

/* MULT.RC.VE's `src`: an LR holds an index into Ra (R0 ++ R1), a CR holds the
 * scalar in its low byte.  Mirrors Ipu._mult_resolve_lcr_scalar. */
static etiss_uint8 mult_resolve_lcr_scalar(IPU *c, etiss_uint32 src)
{
    if (src < IPU_LR_COUNT)
    {
        /* The LR index is read LIVE; only the Ra *data* comes from the
         * snapshot, so a same-cycle LDR_MULT_REG is not yet visible. */
        etiss_uint32 idx = c->LR[src] % (2u * IPU_R_SIZE);
        return c->s_R[idx];
    }
    return (etiss_uint8)(c->CR[src - IPU_LR_COUNT] & 0xFF);
}

void ipu_mult_rc_vv(IPU *c, ETISS_System *sys, etiss_uint32 rc_idx, const etiss_uint8 *ra, etiss_uint32 mask_offset,
                    etiss_uint32 mask_shift, etiss_uint32 cr_idx)
{
    etiss_uint8 rc[IPU_R_SIZE];
    int is_float;
    unsigned i;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    cyclic_read(c->s_R_CYCLIC, rc_idx, rc, IPU_R_SIZE);
    for (i = 0; i < IPU_LANES; ++i)
        lane_store(c->MULT_RES, i, ipu_mult(rc[i], ra[i], (int)c->dtype), is_float);
    mult_mask_and_shift(c, sys, mask_offset, mask_shift, c->CR[cr_idx]);
}

void ipu_mult_rc_ve(IPU *c, ETISS_System *sys, etiss_uint32 rc_idx, etiss_uint32 src, etiss_uint32 mask_offset,
                    etiss_uint32 mask_shift, etiss_uint32 cr_idx)
{
    etiss_uint8 rc[IPU_R_SIZE];
    etiss_uint8 scalar;
    int is_float;
    unsigned i;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    scalar = mult_resolve_lcr_scalar(c, src);
    cyclic_read(c->s_R_CYCLIC, rc_idx, rc, IPU_R_SIZE);
    for (i = 0; i < IPU_LANES; ++i)
        lane_store(c->MULT_RES, i, ipu_mult(rc[i], scalar, (int)c->dtype), is_float);
    mult_mask_and_shift(c, sys, mask_offset, mask_shift, c->CR[cr_idx]);
}

void ipu_mult_rc_vs(IPU *c, ETISS_System *sys, etiss_uint32 rc_idx, etiss_uint32 mask_offset, etiss_uint32 mask_shift,
                    etiss_uint32 cr_idx)
{
    etiss_uint8 rc[IPU_R_SIZE];
    int is_float;
    unsigned i;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    cyclic_read(c->s_R_CYCLIC, rc_idx, rc, IPU_R_SIZE);
    for (i = 0; i < IPU_LANES; ++i)
        lane_store(c->MULT_RES, i, ipu_mult(rc[i], rc[i], (int)c->dtype), is_float);
    mult_mask_and_shift(c, sys, mask_offset, mask_shift, c->CR[cr_idx]);
}

void ipu_mult_ve(IPU *c, ETISS_System *sys, etiss_uint32 ra_idx, etiss_uint32 cr_idx, etiss_uint32 mask_offset,
                 etiss_uint32 mask_shift, etiss_uint32 dstructure_cr_idx)
{
    etiss_uint8 scalar;
    int is_float;
    unsigned i;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    scalar = (etiss_uint8)(c->CR[cr_idx] & 0xFF);
    for (i = 0; i < IPU_LANES; ++i)
    {
        /* Ra is the R0 ++ R1 pair read as one 256-byte cyclic window. */
        etiss_uint32 pos = (ra_idx + i) % (2u * IPU_R_SIZE);
        lane_store(c->MULT_RES, i, ipu_mult(c->s_R[pos], scalar, (int)c->dtype), is_float);
    }
    mult_mask_and_shift(c, sys, mask_offset, mask_shift, c->CR[dstructure_cr_idx]);
}

void ipu_mult_ee(IPU *c, ETISS_System *sys, etiss_uint32 ra_idx, etiss_uint32 cr_idx, etiss_uint32 mask_offset,
                 etiss_uint32 mask_shift, etiss_uint32 dstructure_cr_idx)
{
    etiss_uint8 scalar, ra_byte;
    double result;
    int is_float;
    unsigned i;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    scalar = (etiss_uint8)(c->CR[cr_idx] & 0xFF);
    ra_byte = c->s_R[ra_idx % (2u * IPU_R_SIZE)];
    result = ipu_mult(ra_byte, scalar, (int)c->dtype); /* one product, broadcast */
    for (i = 0; i < IPU_LANES; ++i)
        lane_store(c->MULT_RES, i, result, is_float);
    mult_mask_and_shift(c, sys, mask_offset, mask_shift, c->CR[dstructure_cr_idx]);
}

/* ======================================================================== */
/* ACC slot                                                                 */
/* ======================================================================== */

static void store_mult_res_row_in_acc(IPU *c)
{
    memcpy(c->R_ACC, c->MULT_RES, IPU_R_ACC_SIZE);
}

void ipu_acc_add(IPU *c, ETISS_System *sys)
{
    int is_float;
    unsigned i;
    (void)sys;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    for (i = 0; i < IPU_LANES; ++i)
    {
        double a = lane_load(c->s_R_ACC, i, is_float);
        double m = lane_load(c->MULT_RES, i, is_float);
        lane_store(c->R_ACC, i, ipu_add(a, m, (int)c->dtype), is_float);
    }
}

void ipu_acc_add_first(IPU *c, ETISS_System *sys)
{
    (void)sys;
    IPU_GUARD(c);
    store_mult_res_row_in_acc(c);
}

void ipu_acc_max(IPU *c, ETISS_System *sys)
{
    int is_float;
    unsigned i;
    (void)sys;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    for (i = 0; i < IPU_LANES; ++i)
    {
        double a = lane_load(c->s_R_ACC, i, is_float);
        double m = lane_load(c->MULT_RES, i, is_float);
        /* Python's max(a, m) returns m only when m > a, which is what makes
         * the NaN-carrying case differ from a naive fmax. */
        lane_store(c->R_ACC, i, (m > a) ? m : a, is_float);
    }
}

void ipu_acc_max_first(IPU *c, ETISS_System *sys)
{
    (void)sys;
    IPU_GUARD(c);
    store_mult_res_row_in_acc(c);
}

void ipu_acc_sub(IPU *c, ETISS_System *sys)
{
    int is_float;
    unsigned i;
    (void)sys;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    for (i = 0; i < IPU_LANES; ++i)
    {
        double a = lane_load(c->s_R_ACC, i, is_float);
        double m = lane_load(c->MULT_RES, i, is_float);
        lane_store(c->R_ACC, i, ipu_sub(a, m, (int)c->dtype), is_float);
    }
}

void ipu_acc_sub_first(IPU *c, ETISS_System *sys)
{
    int is_float;
    unsigned i;
    (void)sys;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);
    for (i = 0; i < IPU_LANES; ++i)
    {
        /* 0 - x, not -x: for x = +0.0 that keeps the sign as Python does. */
        double m = lane_load(c->MULT_RES, i, is_float);
        lane_store(c->R_ACC, i, ipu_sub(0.0, m, (int)c->dtype), is_float);
    }
}

void ipu_acc_stride(IPU *c, ETISS_System *sys, etiss_uint32 elements_in_row, etiss_uint32 horizontal_stride,
                    etiss_uint32 vertical_stride, etiss_uint32 offset)
{
    unsigned elements_per_row, num_rows, effective_row_len, num_rows_after_h;
    unsigned after_h[IPU_LANES];
    unsigned n_after_h = 0;
    unsigned out_indices[IPU_LANES];
    unsigned n_out = 0;
    int h_enabled, h_inverted, v_enabled, v_inverted;
    unsigned base, i, row;
    int is_float;

    (void)sys;
    IPU_GUARD(c);
    is_float = lanes_are_float(c);

    /* The Python decode tables have no entry for the reserved encodings, so
     * an out-of-range operand raises there rather than decoding to something. */
    if (elements_in_row > IPU_ELEMENTS_IN_ROW_MAX || horizontal_stride > IPU_STRIDE_MAX ||
        vertical_stride > IPU_STRIDE_MAX)
    {
        ipu_raise(c, IPU_ERR_STRIDE_OPERAND,
                  ((etiss_uint64)elements_in_row << 16) | ((etiss_uint64)horizontal_stride << 8) | vertical_stride);
        return;
    }

    elements_per_row = ipu_elements_per_row(elements_in_row);
    num_rows = IPU_LANES / elements_per_row;
    h_enabled = ipu_stride_enabled(horizontal_stride);
    h_inverted = ipu_stride_inverted(horizontal_stride);
    v_enabled = ipu_stride_enabled(vertical_stride);
    v_inverted = ipu_stride_inverted(vertical_stride);

    if (!h_enabled)
    {
        for (i = 0; i < IPU_LANES; ++i)
            after_h[n_after_h++] = i;
        effective_row_len = elements_per_row;
    }
    else
    {
        unsigned half = elements_per_row / 2, j;
        for (row = 0; row < num_rows; ++row)
        {
            unsigned row_base = row * elements_per_row;
            for (j = 0; j < half; ++j)
                after_h[n_after_h++] = h_inverted ? (row_base + 1 + 2 * j) : (row_base + 2 * j);
        }
        effective_row_len = half;
    }

    num_rows_after_h = effective_row_len ? (n_after_h / effective_row_len) : 0;
    if (!v_enabled)
    {
        for (i = 0; i < n_after_h; ++i)
            out_indices[n_out++] = after_h[i];
    }
    else
    {
        for (row = v_inverted ? 1u : 0u; row < num_rows_after_h; row += 2)
        {
            unsigned start = row * effective_row_len, j;
            for (j = 0; j < effective_row_len; ++j)
                out_indices[n_out++] = after_h[start + j];
        }
    }

    base = (offset % 4u) * 32u;
    for (i = 0; i < n_out; ++i)
        lane_store(c->R_ACC, base + i, lane_load(c->MULT_RES, out_indices[i], is_float), is_float);
}

void ipu_acc_reshape(IPU *c, ETISS_System *sys, etiss_uint32 source, etiss_uint32 dest, etiss_uint32 reshape_mask)
{
    etiss_uint8 src_bytes[8], dst_bytes[8];
    etiss_uint32 mask;
    etiss_uint32 words[IPU_RESHAPE_ELEMENT_COUNT];
    etiss_uint8 targets[IPU_RESHAPE_ELEMENT_COUNT];
    unsigned n = 0, i;

    (void)sys;
    IPU_GUARD(c);
    lrd_get(c->s_LR, source, src_bytes);
    lrd_get(c->s_LR, dest, dst_bytes);

    mask = (reshape_mask >= IPU_RESHAPE_MASK_LR_OFFSET)
               ? c->LR[reshape_mask - IPU_RESHAPE_MASK_LR_OFFSET]
               : reshape_mask;
    if (mask > IPU_RESHAPE_ELEMENT_COUNT)
    {
        ipu_raise(c, IPU_ERR_RESHAPE_MASK_RANGE, mask);
        return;
    }

    /* Gather first, then scatter: every read comes from the pre-instruction
     * snapshot, so overlapping source/dest indices behave as in Python. */
    for (i = mask; i < IPU_RESHAPE_ELEMENT_COUNT; ++i)
    {
        if (src_bytes[i] >= IPU_LANES || dst_bytes[i] >= IPU_LANES)
        {
            ipu_raise(c, IPU_ERR_RESHAPE_INDEX_RANGE, i);
            return;
        }
        memcpy(&words[n], c->s_MULT_RES + (size_t)src_bytes[i] * 4, 4);
        targets[n] = dst_bytes[i];
        ++n;
    }
    for (i = 0; i < n; ++i)
        memcpy(c->R_ACC + (size_t)targets[i] * 4, &words[i], 4);
}

/* -- aggregation ---------------------------------------------------------- */

static unsigned agg_active_lane_count(etiss_uint32 valid_elements)
{
    return valid_elements < IPU_LANES ? (unsigned)valid_elements : IPU_LANES;
}

/* Summed one lane at a time, in lane order: floating-point addition is not
 * associative, so any reassociation would change the answer. */
static double agg_sum_lanes(const etiss_uint8 *buf, unsigned active, int is_float)
{
    unsigned i;
    if (is_float)
    {
        double total = 0.0;
        for (i = 0; i < active; ++i)
            total += lane_load(buf, i, 1);
        return total;
    }
    {
        etiss_int64 total = 0;
        for (i = 0; i < active; ++i)
            total += (etiss_int64)lane_load(buf, i, 0);
        return (double)total; /* wrapped by the caller, like Python's _to_int32 */
    }
}

static double agg_max_lanes(const etiss_uint8 *buf, unsigned active, double seed, int is_float)
{
    double best = seed;
    unsigned i;
    for (i = 0; i < active; ++i)
    {
        double v = lane_load(buf, i, is_float);
        if (v > best)
            best = v;
    }
    return best;
}

void ipu_agg_sum_first(IPU *c, ETISS_System *sys, etiss_uint32 dest_slot, etiss_uint32 cr_idx)
{
    DStructure ds;
    unsigned active, dest;
    int is_float;
    double result;
    (void)sys;
    IPU_GUARD(c);
    if (ipu_decode_dstructure(c, c->CR[cr_idx], &ds) != 0)
        return;
    is_float = lanes_are_float(c);
    active = agg_active_lane_count(ds.valid_elements);
    result = agg_sum_lanes(c->MULT_RES, active, is_float);
    dest = dest_slot % (IPU_R_ACC_SIZE / 4);
    lane_store(c->R_ACC, dest, result, is_float);
}

void ipu_agg_sum(IPU *c, ETISS_System *sys, etiss_uint32 dest_slot, etiss_uint32 cr_idx)
{
    DStructure ds;
    unsigned active, dest;
    int is_float;
    double partial, snap_dest, result;
    (void)sys;
    IPU_GUARD(c);
    if (ipu_decode_dstructure(c, c->CR[cr_idx], &ds) != 0)
        return;
    is_float = lanes_are_float(c);
    active = agg_active_lane_count(ds.valid_elements);
    dest = dest_slot % (IPU_R_ACC_SIZE / 4);
    snap_dest = lane_load(c->s_R_ACC, dest, is_float);
    partial = agg_sum_lanes(c->MULT_RES, active, is_float);
    if (is_float)
        result = partial + snap_dest;
    else
        /* The partial is wrapped to int32 first, then added with wrap -- the
         * two-step rounding Python's _to_int32 + ipu_add(INT8) performs. */
        result = ipu_add((double)ipu_wrap_int32(partial), snap_dest, IPU_DTYPE_INT8);
    lane_store(c->R_ACC, dest, result, is_float);
}

void ipu_agg_max_first(IPU *c, ETISS_System *sys, etiss_uint32 dest_slot, etiss_uint32 cr_idx)
{
    DStructure ds;
    unsigned active, dest;
    int is_float;
    double seed, result;
    (void)sys;
    IPU_GUARD(c);
    if (ipu_decode_dstructure(c, c->CR[cr_idx], &ds) != 0)
        return;
    is_float = lanes_are_float(c);
    active = agg_active_lane_count(ds.valid_elements);
    /* With no active lanes the identity seed is written, so the destination
     * is always defined. */
    seed = is_float ? -INFINITY : -2147483648.0;
    result = agg_max_lanes(c->MULT_RES, active, seed, is_float);
    dest = dest_slot % (IPU_R_ACC_SIZE / 4);
    lane_store(c->R_ACC, dest, result, is_float);
}

void ipu_agg_max(IPU *c, ETISS_System *sys, etiss_uint32 dest_slot, etiss_uint32 cr_idx)
{
    DStructure ds;
    unsigned active, dest;
    int is_float;
    double result;
    (void)sys;
    IPU_GUARD(c);
    if (ipu_decode_dstructure(c, c->CR[cr_idx], &ds) != 0)
        return;
    is_float = lanes_are_float(c);
    active = agg_active_lane_count(ds.valid_elements);
    dest = dest_slot % (IPU_R_ACC_SIZE / 4);
    result = agg_max_lanes(c->MULT_RES, active, lane_load(c->s_R_ACC, dest, is_float), is_float);
    lane_store(c->R_ACC, dest, result, is_float);
}

/* ======================================================================== */
/* AAQ slot                                                                 */
/* ======================================================================== */

void ipu_activate_quantize(IPU *c, ETISS_System *sys, etiss_uint32 activation_fn, etiss_uint32 cr_idx)
{
    DStructure ds;
    unsigned active, i;
    (void)sys;
    IPU_GUARD(c);
    if (c->dtype != IPU_DTYPE_INT8)
    {
        ipu_raise(c, IPU_ERR_ACTIVATE_REQUIRES_INT8, c->dtype);
        return;
    }
    if (ipu_decode_dstructure(c, c->CR[cr_idx], &ds) != 0)
        return;
    active = agg_active_lane_count(ds.valid_elements);

    /* R_ACC is read live and left unmodified; POST_AAQ_REG is fully rewritten
     * with the quantized bytes in front and zeros behind. */
    memset(c->POST_AAQ_REG, 0, IPU_POST_AAQ_REG_SIZE);
    for (i = 0; i < active; ++i)
    {
        double raw = lane_load(c->R_ACC, i, 0);
        double y = ipu_apply_activation((int)activation_fn, raw, c->elu_alpha);
        long q;
        if (y != y) /* NaN: Python's int(round(nan)) raises; 0 keeps us running */
            q = 0;
        else
            q = (long)ipu_py_round(y);
        if (q < -128)
            q = -128;
        else if (q > 127)
            q = 127;
        c->POST_AAQ_REG[i] = (etiss_uint8)(q & 0xFF);
    }
}

/* ======================================================================== */
/* COND slot                                                                */
/* ======================================================================== */

static etiss_int32 to_signed_reg(etiss_uint32 v) { return (etiss_int32)v; }

static void branch_to(IPU *c, etiss_uint32 label)
{
    c->cpu.nextPc = IPU_IMEM_BASE + (etiss_uint64)label * IPU_WORD_BYTES;
}

void ipu_beq(IPU *c, ETISS_System *sys, etiss_uint32 reg1, etiss_uint32 reg2, etiss_uint32 label)
{
    (void)sys;
    IPU_GUARD(c);
    if (reg1 == reg2)
        branch_to(c, label);
}

void ipu_bne(IPU *c, ETISS_System *sys, etiss_uint32 reg1, etiss_uint32 reg2, etiss_uint32 label)
{
    (void)sys;
    IPU_GUARD(c);
    if (reg1 != reg2)
        branch_to(c, label);
}

void ipu_blt(IPU *c, ETISS_System *sys, etiss_uint32 reg1, etiss_uint32 reg2, etiss_uint32 label)
{
    (void)sys;
    IPU_GUARD(c);
    if (to_signed_reg(reg1) < to_signed_reg(reg2))
        branch_to(c, label);
}

void ipu_bge(IPU *c, ETISS_System *sys, etiss_uint32 reg1, etiss_uint32 reg2, etiss_uint32 label)
{
    (void)sys;
    IPU_GUARD(c);
    if (to_signed_reg(reg1) >= to_signed_reg(reg2))
        branch_to(c, label);
}

void ipu_br(IPU *c, ETISS_System *sys, etiss_uint32 reg)
{
    (void)sys;
    IPU_GUARD(c);
    branch_to(c, reg);
}

void ipu_bkpt(IPU *c, ETISS_System *sys)
{
    (void)sys;
    IPU_GUARD(c);
    /* PC = INST_MEM_SIZE halts, exactly as in Python. */
    c->cpu.nextPc = IPU_IMEM_END;
}

/* ======================================================================== */
/* BREAK slot                                                               */
/* ======================================================================== */

int ipu_break(IPU *c, ETISS_System *sys)
{
    (void)c;
    (void)sys;
    return 1;
}

int ipu_break_ifeq(IPU *c, ETISS_System *sys, etiss_uint32 reg, etiss_uint32 value)
{
    (void)c;
    (void)sys;
    return reg == value;
}
