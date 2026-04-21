"""Generate test data for the layernorm_256x144 app.

The kernel runs in INT8 mode (required by the aaq instruction).
Inputs are FP8_E4M3-encoded bytes, treated as raw integer bytes by
INT8 arithmetic.  All results are garbage — the reference replicates
the exact same garbage for bit-exact matching with the emulator.

The 5 aaq truncation points are marked TODO(fp8_aaq).

Output layout: (N_CH*N_TG) rows × N_TPG int32 words = 512 bytes/row.
  Row index for (ch, tg) = ch*2 + tg.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

N_CH  = 144
N_TG  = 2
N_TPG = 128   # tokens per group (SIMD width)

DTYPE = DType.INT8   # run mode required by aaq


def _fp8_one() -> int:
    return int.from_bytes(fp32_to_fp8_bytes(np.array([1.0],  dtype=np.float32), DType.E4), "little")

def _fp8_neg_one() -> int:
    return int.from_bytes(fp32_to_fp8_bytes(np.array([-1.0], dtype=np.float32), DType.E4), "little")


# ---------------------------------------------------------------------------
# Emulator operation replicas
# ---------------------------------------------------------------------------

def _agg_sum_value(acc_vals: list[int]) -> int:
    """Replicate execute_agg(sum, value) in INT8 mode.

    INT8 fmt='<i': result = sum of all acc words, stored as int32 bits → uint32.
    Sum of 128 lanes each ≤ 144×127² ≈ 2.3M → total ≤ 297M, fits in int32.
    """
    raw = sum(acc_vals)
    return struct.unpack("<I", struct.pack("<i", raw))[0]


def _agg_sum_inv_sqrt_int8(acc_vals: list[int]) -> int:
    """Replicate execute_agg(sum, inv_sqrt) in INT8 mode.

    INT8 early-return: f = float(sum), result = 1/sqrt(f), stored as float32 bits.
    """
    f = float(sum(acc_vals))
    result = 1.0 / (f ** 0.5) if f > 0 else 0.0
    return struct.unpack("<I", struct.pack("<f", result))[0]


def _aaq_quantize(acc_int32: int) -> int:
    """Replicate execute_aaq on one accumulator word (INT8 mode).

    val >> 24 (arithmetic), clamped to [-128, 127], returned as uint8.
    """
    signed = struct.unpack("<i", struct.pack("<I", acc_int32 & 0xFFFFFFFF))[0]
    return max(-128, min(127, signed >> 24)) & 0xFF


def _mult_ve_aaq(aaq_raw: int, rc_byte: int) -> int:
    """Replicate mult.ve.aaq for one lane: ipu_mult(aaq&0xFF, rc_byte)."""
    return ipu_mult(aaq_raw & 0xFF, rc_byte, DTYPE)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def _reference_layernorm(
    x_bytes: bytes,
    gamma_bytes: bytes,
    beta_bytes: bytes,
) -> bytes:
    """Replicate the assembly kernel instruction-by-instruction in INT8 mode.

    Mask isolates each token lane during steps 1-3 and phases A/B of step 4.
    Phase C/D (mult.ev) also applies the mask: only the current token's lane
    survives, so each token's output uses its own temp_b[t] value.

    Input layout:   x[ch, tg, tok] at byte (ch*N_TG + tg)*N_TPG + tok
    Output layout:  row (ch*2+tg) at word offset (ch*2+tg)*N_TPG,
                    N_TPG int32 words little-endian.
    """
    one     = _fp8_one()
    neg_one = _fp8_neg_one()

    output = bytearray(N_CH * N_TG * N_TPG * 4)

    for tg in range(N_TG):

        def x_at(ch: int, t: int) -> int:
            return x_bytes[(ch * N_TG + tg) * N_TPG + t]

        # ----------------------------------------------------------------
        # Steps 1–3: per-token stats (aaq0, aaq1, aaq3 per token)
        # The mask zeroes all lanes except t, so agg sees only lane t's value.
        # ----------------------------------------------------------------
        aaq0 = [0] * N_TPG   # mean (uint32 bits)
        aaq1 = [0] * N_TPG   # -mean (uint32 bits)
        aaq3 = [0] * N_TPG   # inv_std (float32 bits as uint32)

        for t in range(N_TPG):
            # Step 1: Σ x[ch,t] × one, accumulate in lane t only
            acc1 = 0
            for ch in range(N_CH):
                acc1 = ipu_add(acc1, ipu_mult(x_at(ch, t), one, DTYPE), DTYPE)
            # agg sees [0, ..., acc1 at t, ..., 0]
            aaq0[t] = _agg_sum_value([acc1 if i == t else 0 for i in range(N_TPG)])
            aaq1[t] = _agg_sum_value([acc1 if i == t else 0 for i in range(N_TPG)])

            # Step 2: Σ x[ch,t]², accumulate in lane t only
            acc2 = 0
            for ch in range(N_CH):
                acc2 = ipu_add(acc2, ipu_mult(x_at(ch, t), x_at(ch, t), DTYPE), DTYPE)
            aaq2_t = _agg_sum_value([acc2 if i == t else 0 for i in range(N_TPG)])

            # Step 3a: mult.ve.aaq aaq0[t] × ones[t]; acc.first; aaq → TEMP_BASE+0
            # TODO(fp8_aaq)
            prod3a   = _mult_ve_aaq(aaq0[t], one)      # scalar × ones[t]
            temp0_3a = _aaq_quantize(prod3a)

            # Step 3b: mean × neg_one; acc.first; aaq → TEMP_BASE+128
            # TODO(fp8_aaq)
            prod3b   = ipu_mult(temp0_3a, neg_one, DTYPE)
            temp1_3b = _aaq_quantize(prod3b)

            # Step 3c: mean × (-mean); acc.first
            prod3c = ipu_mult(temp0_3a, temp1_3b, DTYPE)

            # Step 3d: mult.ve.aaq aaq2 × ones[t]; acc
            prod3d = _mult_ve_aaq(aaq2_t, one)
            var_t  = ipu_add(prod3c, prod3d, DTYPE)

            # agg sum inv_sqrt
            aaq3[t] = _agg_sum_inv_sqrt_int8([var_t if i == t else 0 for i in range(N_TPG)])

        # ----------------------------------------------------------------
        # Step 4: per-channel affine, all N_TPG tokens in SIMD
        # Dummy iteration (ch_idx=-1) writes to the ch=0 output row and is
        # overwritten by the real ch=0 pass — skip it in the reference.
        # ----------------------------------------------------------------
        for ch in range(N_CH):
            # Phase A: x[ch,t]-mean, per token t (mask isolates lane t)
            # aaq truncation #3 // TODO(fp8_aaq)
            temp_a = [0] * N_TPG
            for t in range(N_TPG):
                prod  = ipu_mult(x_at(ch, t), one, DTYPE)
                # acc.add_aaq.first aaq1[t]: r_acc[t] = prod + aaq1[t] (raw uint32)
                c1    = ipu_add(prod, aaq1[t], DTYPE)
                temp_a[t] = _aaq_quantize(c1)

            # Phase B: (x-mean) × inv_std, per token t
            # mult.ve.aaq aaq3[t]: scalar=aaq3[t]&0xFF × r_cyclic[t]=temp_a[t]; mask → lane t
            # aaq truncation #4 // TODO(fp8_aaq)
            temp_b = [0] * N_TPG
            for t in range(N_TPG):
                prod   = _mult_ve_aaq(aaq3[t], temp_a[t])
                temp_b[t] = _aaq_quantize(prod)

            # Phase C: normalized × γ[ch], per token t
            # mult.ev: ra=TEMP_BASE(=temp_b), fixed_cyclic_idx=ch_idx → scalar=γ[ch_idx]=γ[ch]
            # result[i] = ipu_mult(temp_b[i], γ[ch]); mask → only lane t survives
            gam_ch = gamma_bytes[ch]
            acc_c  = [ipu_mult(temp_b[t], gam_ch, DTYPE) for t in range(N_TPG)]

            # Phase D: ones × β[ch], per token t; acc
            bet_ch = beta_bytes[ch]
            acc_d  = [ipu_add(acc_c[t], ipu_mult(one, bet_ch, DTYPE), DTYPE)
                      for t in range(N_TPG)]

            # str_acc_reg stores all N_TPG int32 words for this (ch, tg) row
            out_row = ch * 2 + tg
            for t in range(N_TPG):
                struct.pack_into("<i", output, (out_row * N_TPG + t) * 4, acc_d[t])

    return bytes(output)


def main() -> None:
    out_dir  = Path(__file__).parent / "test_data_format"
    dtype_dir = out_dir / "fp8_e4m3"
    dtype_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    x_fp32     = rng.uniform(-1.0, 1.0, size=(N_CH, N_TG, N_TPG)).astype(np.float32)
    gamma_fp32 = rng.uniform(0.5,  1.5,  size=N_CH).astype(np.float32)
    beta_fp32  = rng.uniform(-0.5, 0.5,  size=N_CH).astype(np.float32)

    x_bytes     = fp32_to_fp8_bytes(x_fp32.reshape(-1), DType.E4)
    gamma_bytes = fp32_to_fp8_bytes(gamma_fp32,          DType.E4)
    beta_bytes  = fp32_to_fp8_bytes(beta_fp32,           DType.E4)

    print("Generating layernorm_256x144 test data...")
    golden = _reference_layernorm(x_bytes, gamma_bytes, beta_bytes)

    (dtype_dir / "input_fp8_e4m3.bin").write_bytes(x_bytes)
    (dtype_dir / "gamma_fp8_e4m3.bin").write_bytes(gamma_bytes)
    (dtype_dir / "beta_fp8_e4m3.bin").write_bytes(beta_bytes)
    (dtype_dir / "out_fp8_e4m3_acc_int32.bin").write_bytes(golden)

    print(f"  [fp8_e4m3] x={len(x_bytes)}B  gamma={len(gamma_bytes)}B  "
          f"beta={len(beta_bytes)}B  golden={len(golden)}B")
    print("Done.")


if __name__ == "__main__":
    main()
