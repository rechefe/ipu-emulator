# IPU conformance suite

The expected behaviour of the ISA, as data (E04-31, Q60). Any model (the
Python emulator, the RTL under cocotb, a future ETISS model) runs the same
cases and compares its trace with the expected one, record by record. No case
contains Python: a consumer needs only these files and the schemas.

## A case

Each folder is one case:

| File | What it holds |
|---|---|
| `program.asm` | The source, for reading. When `program.bin` holds a word the assembler does not produce, the second line says how the assembled words were edited. |
| `program.bin` | The assembled words, what a model loads: `ipu-as` program-file format, each word little-endian. |
| `setup.json` | The starting state and run limits (`setup.schema.json`). |
| `expected.trace.jsonl` | The trace of the run: a header line (`ipu_isa/schemas/trace.v1.json`), then one commit record per word (`commit_record.v1.json`). |

The case names say what they cover:

- `insn_<mnemonic>_<format>`: every HARDWARE instruction, in INT8 and E4, and
  in FP32 for the datapath slots (load, store, mult, acc, aaq), from a seeded
  random state;
- `probe_<name>_<format>`: the timing probes, which show which value of a
  register an operand reads (cycle start or forwarded);
- `fault_<id>_*`: each modelled fault (F-1 to F-6, F-9), and F-1 and F-9 with
  their enable bit clear;
- `arith_*`: the arithmetic events (accumulator overflow, NaN result).

A case with `"emulator_only": true` uses a wide format (FP32, INT32) or an
EMULATOR slot's instruction, which the hardware does not have; an RTL
testbench skips it.

## Running a case

1. Reset the model in `setup.json`'s `format`.
2. Load `program.bin`; the PC starts at 0.
3. Write each of `registers` (`name` and `index` as `ipu_isa.registers` names
   them; `value` for `lr` and `cr`, `hex` bytes for a vector register), then
   each `xmem` region. Every other register holds its reset value and every
   other XMEM byte is zero.
4. Set each of `fault_enables` (every other fault stays enabled).
5. Run until a `HALT` word, a fault, or `max_cycles` words (fault F-9; 0
   disables the limit). A `BREAK` slot never stops the run.
6. Compare the trace: the header's `schema_version`, `isa_digest`, `format`
   and `program_digest` (SHA-256 of `program.bin`), not `emulator_version`;
   then every record, field by field. A record's `cycle` counts the records
   before it.

The Python emulator's runner is
`src/tools/ipu-emu-py/test/conformance.py`, through the reference-model API
(`ipu_emu.reference`); `bazel test //src/tools/ipu-emu-py:test_conformance`
runs every case and names the first differing cycle and field.

## Regenerating

```bash
bazel run //src/tools/ipu-emu-py:regen_conformance
```

It rewrites every case from `src/tools/ipu-emu-py/test/conformance_cases.py`
and removes the folders of cases no longer listed. Review the diff as a
golden's: a behaviour-change PR regenerates only the traces it changes and
says why each changed. A change to the ISA changes every trace header's
`isa_digest`.
