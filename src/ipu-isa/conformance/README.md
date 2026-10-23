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
| `program.bin` | The assembled words, what a model loads: 24 bytes per word, little-endian, no file header (the `ipu-as` program-file format). |
| `setup.json` | The starting state and run limits (`setup.schema.json`). |
| `expected.trace.jsonl` | The trace of the run: a header line (`ipu_isa/schemas/trace.v1.json`), then one commit record per word (`commit_record.v1.json`). |

The case names say what they cover:

- `insn_<mnemonic>_<format>`: every HARDWARE instruction, in INT8 and E4, and
  in FP32 for the datapath slots (load, store, mult, acc, aaq), from a seeded
  random state. `ACTIVATE.QUANTIZE` has no E4 case: it quantizes to INT8
  only (OQ-20);
- `probe_<name>_<format>`: the timing probes, which show which value of a
  register an operand reads (cycle start or forwarded), but three: `idle_word`
  (an empty IMEM entry has no program-file encoding), `break_ifeq` and
  `break_resume` (a run never evaluates `BREAK`);
- `fault_<id>_*`: each modelled fault (F-1 to F-6, F-9), and F-1 and F-9 with
  their enable bit clear;
- `arith_*`: the arithmetic events (accumulator overflow, NaN result).

A case with `"emulator_only": true` uses a wide format (FP32, INT32) or an
EMULATOR slot's instruction, which the hardware does not have; an RTL
testbench skips it.

## Running a case

1. Reset the model in `setup.json`'s `format`.
2. Load `program.bin` into IMEM from word 0; the PC starts at 0. The program
   length is the number of words in the file: fetching at a PC at or past it
   faults F-2.
3. After the reset, zero every register and every XMEM byte, except `CR0`
   (0) and `CR1` (1), the constants. Write each of `registers`, then each `xmem`
   region. A register is named by its bank and `index` (the register within
   the bank, table below); `value` is an `lr` or `cr` register's unsigned
   32-bit value, `hex` a vector register's bytes, lowest address first. A
   register whose reset value is not zero, such as `CR15` (the default
   dstructure) or `r_mask`, is always listed.
4. Set each of `fault_enables` (every other fault stays enabled).
5. Run until a `HALT` word, a fault, or the cycle limit: after `max_cycles`
   words, the next word faults F-9 before it runs (its record lists no
   slots); 0 disables the limit. A `BREAK` slot never stops the run. The
   activation parameters keep their defaults, and in the wide formats
   `ACTIVATE.QUANTIZE` applies the activation without quantizing.
6. Compare the trace: the header's `schema_version`, `isa_digest`, `format`
   and `program_digest` (SHA-256 of `program.bin`), not `emulator_version`;
   then every record, field by field. A record's `cycle` counts the records
   before it.

The register banks:

| Bank | Registers | Bytes each |
|---|---|---|
| `r` | 2 (`R0`, `R1`) | 128 |
| `r_wide_debug` | 2 | 512 |
| `r_cyclic` | 1 | 512 |
| `r_cyclic_wide_debug` | 1 | 2048 |
| `r_meta` | 2 (`R0`, `R1` metadata) | 2 |
| `r_cyclic_meta` | 4 (one per `R_CYCLIC` slot) | 2 |
| `r_mask` | 1 | 128 |
| `r_acc` | 1 | 512 |
| `post_aaq_reg` | 1 | 512 |
| `aaq_lut` | 1 (LUT table and metadata) | 550 |
| `lr` | 16 (`LR0`-`LR15`) | 4 |
| `cr` | 16 (`CR0`-`CR15`) | 4 |
| `mult_res` | 1 | 512 |
| `mem_bypass` | 1 | 128 |

The `_wide_debug` banks hold the wide formats' operands (emulator only). `r_meta`,
`r_cyclic_meta` and `aaq_lut` hold row metadata and the AaQ LUT (ADR 0007).

The Python emulator's runner is
`src/ipu-emu/test/conformance.py`, through the reference-model API
(`ipu_emu.reference`); `bazel test //src/ipu-emu:test_conformance`
runs every case and names the first differing cycle and field.

## Regenerating

```bash
bazel run //src/ipu-emu:update_conformance
```

It rewrites every case from `src/ipu-emu/test/conformance_cases.py`
and removes the folders of cases no longer listed. Review the diff as a
golden's: a behaviour-change PR regenerates only the traces it changes and
says why each changed. A change to the ISA changes every trace header's
`isa_digest`.
