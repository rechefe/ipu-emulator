# Emulator faults follow the hardware fault list

_Status: accepted (2026-10-15). Implementation is planned in `planning/architecture-deepening/`. Amended by [ADR 0007](0007-operational-aaq-and-nx8.md): invalid QF, DS and MULT constant contents are configuration errors, raised in the word's check phase; the fault list is unchanged._

The emulator reports exactly the faults F-1 to F-9 from `docs/content/requirements/faults.md`, each with its ID and an enable bit: enabled faults halt and record the fault code and pc, disabled faults set a sticky flag; all are enabled at reset. F-8 (word stall limit) is listed as not modelled because the emulator has no stall model. Accumulator overflow and NaN are arithmetic events: IEEE results plus a sticky flag, never a halt.

The only other error kind is a configuration error, raised before the first cycle. There is no emulator-only "simulation limit": the cycle limit is fault F-9.
