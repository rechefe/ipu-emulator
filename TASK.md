

Plan
1) Define a shared instruction spec module
	- Create src/tools/ipu-common/ipu_spec/ with InstructionDef, OperandDef, InstructionDoc, and a registry.
	- Move authoritative opcode lists and operand definitions there.
	- Add validation: unique opcode names, stable opcode order, unique operand names per opcode.

2) Wire the assembler to the shared spec
	- Update ipu_as.opcodes to read from the shared registry (no local opcode lists).
	- Update ipu_as.inst to build struct_by_opcode_table from the shared spec.
	- Update CompoundInst.get_fields to generate fields from the shared spec, not class names.

3) Add operand-name access in the emulator
	- Extend decode to return both raw fields and operand-name keyed fields.
	- In execute.py, read operands via InstructionDoc operand names (e.g. inst.fields["LrOffset"]).
	- Replace hardcoded opcode indices with shared spec enums or registry lookup.

4) Keep register definitions in one place
	- Derive LR/CR/R counts from the shared spec or REGFILE_SCHEMA to remove duplication.
	- Use shared constants in both assembler and emulator.

5) Add compatibility checks
	- Add a small test that assembles a program, decodes it, and verifies operand-name keys.
	- Add a startup assert in the emulator that opcode order matches the spec.

Deliverables
- Shared instruction spec module with validation.
- Assembler and emulator consuming the same opcode/operand definitions.
- Emulator field access by InstructionDoc operand names.
- Basic tests/validation for spec consistency.