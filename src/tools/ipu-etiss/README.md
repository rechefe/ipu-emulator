# `ipu-etiss` — the IPU as a native ETISS architecture

This package makes the IPU a first-class architecture for
[ETISS](https://github.com/tum-ei-eda/etiss), the Extendable Translating
Instruction Set Simulator. An ETISS `CPUCore` executes assembled IPU programs
directly: each VLIW word is one ETISS instruction, translated to C and compiled
by ETISS's JIT.

The Python emulator (`ipu_emu`) remains the reference implementation. This
backend is checked against it by differential tests that compare the whole
machine state, not just program output.

Design notes: [`docs/content/specs/etiss-integration.md`](../../../docs/content/specs/etiss-integration.md).

## Layout

| Path | What it is |
|------|------------|
| `arch/IPUArch.{h,cpp}` | The `etiss::CPUArch` implementation: CPU state, reset, instruction set registration, `VirtualStruct` for debuggers |
| `arch/IPUArchLib.cpp` | ETISS plugin entry points |
| `arch/IPUFuncs.c` | Instruction semantics — a C port of `ipu_emu/ipu.py` |
| `runtime/ipu_math.{c,h}` | FP8 `e(x)m(8-x)` codec and INT8 arithmetic, ported from `ipu_math.py` |
| `runtime/ipu_activations.{c,h}` | The 12 activation functions, ported from `activations.py` |
| `runtime/test/` | Bit-exactness harness for the two runtime modules |
| `runner/ipu_etiss_run.cpp` | The command-line runner the Python driver invokes |
| `patches/` | Patches applied to ETISS before building (see below) |
| `build_etiss.sh` | Fetches, patches and builds everything |

Nothing here hard-codes an opcode, a bit offset, or a register size. `IPU_gen.h`
(the `struct IPU` and constants), `IPUFuncs_gen.h` (handler prototypes) and
`IPUDecode_gen.cpp` (the decode callback) are generated from
`instruction_spec.py` and `registers.py` by `ipu_as.gen_etiss`, which runs as
part of the CMake build.

## Building

```bash
sudo apt-get install -y cmake build-essential \
    libboost-system-dev libboost-filesystem-dev libboost-program-options-dev

./src/tools/ipu-etiss/build_etiss.sh
export IPU_ETISS_RUN=$PWD/build/etiss/install/bin/ipu_etiss_run
```

## Using it

```python
from ipu_emu.emulator import run_test

state, cycles = run_test(inst_path="prog.bin", setup=..., backend="etiss")
```

`IpuApp.run(backend="etiss")` works the same way, and `$IPU_EMU_BACKEND`
switches the default for a whole test run. The ETISS path is additive: omit the
argument and everything runs on the Python emulator exactly as before.

Parity tests:

```bash
pytest src/tools/ipu-emu-py/test/test_etiss_parity.py    # every instruction
pytest src/tools/ipu-apps/test/test_etiss_app_parity.py  # whole applications
```

Both skip when `$IPU_ETISS_RUN` is unset.

## The ETISS patch

ETISS's instruction fetch carries the fetched bytes into the decoder through
`BitArray::set_value(unsigned long)`, fed by `Buffer::data()`, which returns
only the **first 32-bit word** of the buffer. Every ISA that ships with ETISS
has instructions of 16 or 32 bits, so nothing has hit this; the 186-bit IPU
word was silently truncated, and every slot above bit 31 decoded as garbage.

`patches/0001-wide-instruction-fetch.patch` adds
`BitArray::set_value_bytes(const unsigned char *, unsigned)` and uses it at the
two fetch sites. On a little-endian host it is bit-for-bit what the old path did
for 16- and 32-bit instructions, so existing architectures are unaffected. This
is a candidate to send upstream.

## Scope

Implemented: the complete ISA in the narrow (INT8 / FP8) datapath — every
instruction in `INSTRUCTION_SPEC`, the multiply masking and partition vectors,
`ACC.STRIDE` / `ACC.RESHAPE` / the `AGG` reductions, all 12 activations, and
`RunStats`.

Not implemented: **wide-vector debug mode** (`IpuState(wide_vector_debug=True)`,
4-byte lanes). It is an emulator-only analysis feature with no hardware
counterpart, so the ETISS backend rejects it with a clear error rather than
approximating it. The softmax applications use that mode and therefore run on
the Python backend only.
