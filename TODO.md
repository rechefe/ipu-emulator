# IPU TODOs

- Write basic kernel for softmax - pseudocode
- Write basic kernel for CNN
- Implement quantization
- Add preprocessing CLI options (overriding parameters inside the `asm`)
- Add "Metadata" to registers and XMEM
- Add support for "union" like instruction in the assembler, specifically for LR instruction (immediate and LCR can be shared)

## Bugs
- [Wide-mode R0/R1 mult read uses snapshot, not live](planning/bug-wide-mode-r0r1-snapshot.md) — same-cycle LDR_MULT_REG invisible in wide mode (live is correct)