# CR scalars decoded in the active element format

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-28 · Size S · Depends on: E04-15, E04-17 · Behaviour change: yes (in FP32 and INT32, a CR scalar operand is the CR's low 32 bits in that format instead of its low byte as a signed integer)

## Goal

Apply Q49: a CR used as a scalar element operand is decoded in the active element format, from its low element-width bits. One rule for every format; documented in the ISA reference.

## Background

- Narrow formats already follow this rule: `_mult_resolve_lcr_scalar` takes the CR's low byte (ipu.py:1238), and the multiply decodes it in the active format (INT8 or E1..E7).
- Wide formats do not: `_wide_cr_scalar_byte_as_int32` (ipu.py:693-696) takes the CR's low byte as a signed integer, and FP32 converts it with `float(...)` (ipu.py:1249-1250). So in FP32 a CR value of 1 means 1.0 today; under the new rule it is the float32 bit pattern 0x00000001 (a subnormal).
- The alias observer probes "scalar is one" with the same function (alias_observer.py:225, 230).
- `CR1` is read-only and holds the integer 1 (`CR_READ_ONLY_INITIAL_VALUES`, ipu_config.py:16-19). Kernels may use it as a "one" scalar.
- After E04-15 the rule lives in one place: `LaneCodec.decode_scalar`.

## Implementation

1. **Audit first.** List every kernel and test that uses a CR scalar operand (MULT.VE, MULT.EE, MULT.RC.VE, MULT.RC.VS with a CR source, and any other `CrIdx` element scalar) under FP32 or INT32, with the CR values the host programs. Put the table in the PR.
2. Change `Fp32Codec.decode_scalar` to reinterpret the low 32 bits as float32, and `IntCodec` for INT32 to use the low 32 bits as int32.
3. Update host-side CR programming in the audited apps to write the float32 bit pattern (for example `struct.unpack("<I", struct.pack("<f", 1.0))[0]`). Update the alias observer's identity probe to compare the decoded value with 1.
4. Document the rule in the ISA reference page for CR operands (coordinate with E03/E09 for the page).

## Open question

- `CR1` holds the integer 1. In FP32 that decodes as a subnormal, not 1.0. Should `CR1` hold "one in the active format", or must kernels that need 1.0 program another CR? This is a hardware-contract question (D4) for the user.

## Tests

- [ ] For each format, a MULT.VE with a CR scalar equals the codec's decode of the CR's low element-width bits times the vector.
- [ ] The audited kernels pass with their outputs unchanged after the host-side CR change.
- [ ] Identity-probe statistics unchanged for the audited kernels.

## Acceptance Criteria

- [ ] One scalar decode rule, in the codec, for every format.
- [ ] The audit table and every host-side CR change are in the PR.
- [ ] The rule is written in the ISA reference.
