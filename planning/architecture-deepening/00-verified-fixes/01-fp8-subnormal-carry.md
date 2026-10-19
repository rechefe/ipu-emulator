# FP8 encoder: a subnormal that rounds up must carry into the smallest normal

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-01 · Size S · Depends on: none · Behaviour change: yes (host-side FP8 encoding of values just below the smallest normal now gives the smallest normal instead of the largest subnormal; for E7 such values give the smallest normal instead of zero)

## Goal

`_float32_to_fp8_scalar` rounds every finite input to the nearest representable value of its format (ties to even), for E1 to E7. An exhaustive codec test checks the encoder and decoder against values computed independently of `ipu_math.py`, so the bug cannot come back.

## Background

Verified at 694a8a8 (`src/tools/ipu-emu-py/src/ipu_emu/ipu_math.py`):

- `_fp8_encode_subnormal` (ipu_math.py:88-96) computes `man_int = round(val * 2**(man_bits + bias - 1))` and then clamps it with `max(0, min(man_int, max_man))` (ipu_math.py:95). When a value lies in the upper half of the gap between the largest subnormal and the smallest normal, `man_int` is `2**man_bits`. The correct result is exponent field 1, mantissa 0, which is exactly the integer `2**man_bits`. The clamp returns the largest subnormal instead.
- `_fp8_encode_normal` (ipu_math.py:99-112) handles the same carry correctly (ipu_math.py:106-109). Only the subnormal path is wrong.
- Reproduced at HEAD with `v = (max_subnormal + min_normal) / 2 * 1.0001`:

  | Format | Got | Expected |
  |---|---|---|
  | E2 | `0x1f` (0.96875) | `0x20` (1.0) |
  | E3 | `0x0f` (0.234375) | `0x10` (0.25) |
  | E4 | `0x07` (0.013671875) | `0x08` (0.015625) |
  | E5 | `0x03` | `0x04` |
  | E6 | `0x01` | `0x02` |
  | E7 | `0x00` (0.0) | `0x01` (2^-62) |

  E7 has no mantissa bits, so its only subnormal is zero: every value in `[min_normal / 2, min_normal)` encodes as zero today.
- E1 is different. Its bias is 0, so exponent field 1 is already the NaN exponent and every finite E1 value is a subnormal. There the clamp is correct: a carry must saturate to the largest finite value (`0x3f`), matching the overflow convention (ipu_math.py:144-145).
- The encoder runs only on the host: `fp32_to_fp8_bytes` (ipu_math.py:151-161), used by `load_fp32_as_fp8_to_xmem` (emulator.py:239-262), and `dtype_one_byte` (ipu_math.py:174-181, which encodes 1.0, a normal value). No emulator instruction calls it, so no kernel cycle count or kernel golden changes.
- There is no `ipu_math` unit test. The FP8 tests in `test_execute.py:2144-2300` build their inputs with the encoder under test, so they cannot catch an encoder bug (review register item 19).
- `ml_dtypes` 0.6.0 is already resolved for Bazel (`ipu-emu-py/pyproject.toml:13`, `ml-dtypes>=0.2.0`). Its `float8_e3m4`, `float8_e4m3` and `float8_e5m2` use the same finite encodings as E3, E4 and E5. They differ only at the all-ones exponent (inf/NaN there; NaN only here). `float8_e3m4` and `float8_e4m3` exist from `ml_dtypes` 0.5.0.

## Decisions applied

- Q34: the encoder bug is fixed regardless of the FP8 naming and special-value questions. The special-value convention (NaN at the all-ones exponent, saturation instead of inf) is an open issue and is **not** changed here.
- Q24, Q47: all seven formats E1 to E7 are in scope, not only E4 and E5.

## Implementation

1. In `_fp8_encode_subnormal`, replace the clamp:
   ```python
   man_int = round(val * (2.0 ** (man_bits + bias - 1)))
   if man_int > max_man:
       # Rounded up out of the subnormal range. (1 << man_bits) is exponent
       # field 1, mantissa 0: the smallest normal. E1 has no normal range
       # (exponent field 1 is NaN), so it saturates instead.
       if exp_bits == 1:
           return (sign << 7) | _fp8_max_finite(exp_bits, man_bits)
       man_int = 1 << man_bits
   return (sign << 7) | man_int
   ```
   Keep Python's `round` (round half to even). It is the rounding the normal path already uses.
2. Raise the lower bound in `ipu-emu-py/pyproject.toml` to `ml-dtypes>=0.5.0`, so the test's dtypes exist in any resolved environment.
3. Add `src/tools/ipu-emu-py/test/test_ipu_math.py` and a `py_pytest_test(name = "test_ipu_math")` in `ipu-emu-py/BUILD.bazel` with deps `:ipu_emu_lib`, `requirement("ml-dtypes")` and `requirement("pytest")`.
4. In the test, write an independent reference from the format definition using `fractions.Fraction`. Do not import any private helper from `ipu_math`:
   - `ref_decode(code, e)`: sign bit, `e` exponent bits, `7 - e` mantissa bits, bias `2**(e-1) - 1`; all-ones exponent is NaN; zero exponent is subnormal.
   - `ref_encode(x, e)`: the representable value nearest to `x`, ties to the even mantissa; magnitudes above the largest finite value saturate to it; NaN gives a NaN code.

## Tests

- [ ] Decode, exhaustive: for each of E1 to E7 and each of the 256 codes, `fp8_bytes_to_fp32` equals `ref_decode` exactly (NaN codes give NaN).
- [ ] Encode round trip: every non-NaN code (both zeros included) encodes back to itself from its decoded value.
- [ ] Encode midpoints: for each pair of adjacent representable magnitudes, the exact midpoint goes to the even code, and the midpoint plus or minus one float32 ULP goes to the nearer code. This covers the subnormal-to-normal gap and the normal exponent carries. All midpoints are exact in float32 for E1 to E7.
- [ ] Encode saturation: `max_finite * 1.01`, `inf` and `-inf` give the largest finite code with the right sign. NaN encodes to a code that decodes to NaN.
- [ ] Regression cases from the table above (E2 to E7) and the E1 carry case (`1.99` gives `0x3f`).
- [ ] `fp32_to_fp8_bytes` (the float32 array path) agrees with the scalar path on all inputs above.
- [ ] Cross-check with `ml_dtypes`: for E3, E4 and E5, on all midpoint inputs and 100 000 seeded random float32 values within `±max_finite`, the encoder's byte equals `np.float32(x).astype(ml_dtypes.float8_e<x>m<y>).view(np.uint8)`. Decoding of every code whose exponent is not all ones matches too.
- [ ] `bazel test //...` is green. If a test outside `test_ipu_math` changes, it is one that fed a value from the affected range through the encoder; list it in the PR.

## Acceptance Criteria

- [ ] No E2 to E7 input in `[max_subnormal, min_normal)` encodes to a value farther away than the nearest representable value.
- [ ] `test_ipu_math` exists, runs in the default tier, and uses no encoder output as an expected value.
- [ ] The NaN and saturation convention is unchanged (it stays an open issue under Q34).
- [ ] The PR is labelled as a behaviour change and names the affected range per format.

## Risks

- E04-13 (element-format lane codecs) replaces the scalar encoder with a vectorised codec and proves bit identity against it. That issue depends on this one; land this first so the reference is correct.
