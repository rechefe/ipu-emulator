# Mission: AaQ quantization and XMEM row metadata in the IPU emulator

## Why
Make the IPU emulator's AaQ stage match the hardware spec, including the 16-bit metadata word beside every XMEM row, while keeping today's kernels running. To make the design decisions and review the implementation, the user needs a working model of how quantization, the row metadata and the cache unit's tables fit together.

## Success looks like
- Say, for any piece of information about a row, where it lives (data bytes, metadata word, table configuration, AaQ stage) and who writes it.
- Decode a quantized byte by hand, given its row's scale and its table's format.
- Answer the open design questions (Q7–Q12 of the grilling session) with reasons.
- Point out what the specs leave undefined (for example, how MULT expands a row to FP32).
- Review the emulator PR that implements the hardware AaQ.

## Constraints
- Sources are the repo specs (`stage-aaq-str.md`, `cache-unit.md`, `open-questions.md`) first, industry references second.
- The specs are incomplete; lessons mark where they stop.
- The user works in the claude.ai web app: lessons are published as one artifact.

## Out of scope
- General ML quantization theory beyond what the specs use.
- RTL implementation details.
- The cache unit's DMA, tag and flush machinery, beyond carrying metadata.
