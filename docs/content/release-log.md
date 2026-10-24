# Release Log

This page lists every pull request that has been merged into `master`, newest
first.  The docs workflow refreshes it each time it publishes the site.

| # | Title | Merged at (UTC) |
|---|-------|----------------|
| [#263](https://github.com/rechefe/ipu-emulator/pull/263) | Operational AaQ: NX8 quantization, row metadata and the spec's AaQ instructions | 2026-10-24 07:31 |
| [#262](https://github.com/rechefe/ipu-emulator/pull/262) | AaQ, MULT and ACC stage specs (MAS + ARCH) | 2026-10-23 13:02 |
| [#251](https://github.com/rechefe/ipu-emulator/pull/251) | IPU vscode extension - leftovers settings | 2026-10-19 13:23 |
| [#255](https://github.com/rechefe/ipu-emulator/pull/255) | Architecture deepening: ISA package, emulator core, assembler, kernel framework, tooling and docs (E00–E09) | 2026-10-19 04:35 |
| [#252](https://github.com/rechefe/ipu-emulator/pull/252) | Add IPU hardware requirements documentation | 2026-10-04 17:39 |
| [#249](https://github.com/rechefe/ipu-emulator/pull/249) | Vscode extension update | 2026-09-28 10:55 |
| [#250](https://github.com/rechefe/ipu-emulator/pull/250) | codegen: declare SV union operands at their own width, type and name | 2026-09-27 15:48 |
| [#237](https://github.com/rechefe/ipu-emulator/pull/237) | perf(emu): add prototyping mode | 2026-09-24 18:33 |
| [#235](https://github.com/rechefe/ipu-emulator/pull/235) | Add window activation | 2026-09-24 18:33 |
| [#217](https://github.com/rechefe/ipu-emulator/pull/217) | Add a VS Code extension for IPU assembly | 2026-09-24 18:33 |
| [#246](https://github.com/rechefe/ipu-emulator/pull/246) | Yarden&david merge | 2026-09-24 18:32 |
| [#247](https://github.com/rechefe/ipu-emulator/pull/247) | fix: update AaQ stage documentation to clarify dynamic_exponent and f… | 2026-09-24 11:29 |
| [#244](https://github.com/rechefe/ipu-emulator/pull/244) | Aa q spec fix | 2026-09-14 17:52 |
| [#236](https://github.com/rechefe/ipu-emulator/pull/236) | Sandbox Jinja | 2026-09-03 12:37 |
| [#234](https://github.com/rechefe/ipu-emulator/pull/234) | performance improvement. optimize per-cycle execution | 2026-09-03 12:36 |
| [#233](https://github.com/rechefe/ipu-emulator/pull/233) | enhance debugging documentation and add XMEM inspection features | 2026-09-03 12:28 |
| [#231](https://github.com/rechefe/ipu-emulator/pull/231) | implement EmulatorError for handling invalid operations and update CR… | 2026-09-03 12:27 |
| [#229](https://github.com/rechefe/ipu-emulator/pull/229) | extend dstructure configuration handling from CR15 only to all CRs | 2026-09-03 12:25 |
| [#222](https://github.com/rechefe/ipu-emulator/pull/222) | feat: add identity kernel example with Python harness and assembly | 2026-09-03 12:24 |
| [#219](https://github.com/rechefe/ipu-emulator/pull/219) | Correct two r_cyclic docstrings left stale by #180 | 2026-09-03 12:23 |
| [#206](https://github.com/rechefe/ipu-emulator/pull/206) | kernel registry: kernels declare themselves, coverage is generated | 2026-08-23 19:27 |
| [#205](https://github.com/rechefe/ipu-emulator/pull/205) | softmax: benchmarks and a layout contract test | 2026-08-19 14:18 |
| [#204](https://github.com/rechefe/ipu-emulator/pull/204) | softmax: five FP32 wide-vector kernels | 2026-08-19 14:16 |
| [#215](https://github.com/rechefe/ipu-emulator/pull/215) | Fixes issue #210: `ACC.RESHAPE` (previously `RESHAPE`) had three bugs and a missing feature. | 2026-08-16 20:14 |
| [#216](https://github.com/rechefe/ipu-emulator/pull/216) | Adds `partition_mask`, a new 3-bit input signal to the AaQ stage, to the stage specification (`docs/content/specs/stage-aaq-str.md`). It lets the caller mask off whole partition groups from the right, so their elements are excluded from activation/quantization instead of being processed. | 2026-08-16 20:13 |
| [#214](https://github.com/rechefe/ipu-emulator/pull/214) | Refactor documentation and comments to replace "lane" with "element"  | 2026-08-15 19:02 |
| [#209](https://github.com/rechefe/ipu-emulator/pull/209) | docs: add Accumulator Stage Specification to navigation | 2026-08-12 19:36 |
| [#208](https://github.com/rechefe/ipu-emulator/pull/208) | New spec cache | 2026-08-08 18:54 |
| [#196](https://github.com/rechefe/ipu-emulator/pull/196) | mode-blindness: make r_cyclic's rc_idx element-addressed | 2026-08-04 15:34 |
| [#195](https://github.com/rechefe/ipu-emulator/pull/195) | mode-blindness: minor cleanups in acc/agg handlers | 2026-08-04 15:33 |
| [#194](https://github.com/rechefe/ipu-emulator/pull/194) | New spec for aa q str | 2026-08-01 09:02 |
| [#193](https://github.com/rechefe/ipu-emulator/pull/193) | Add RESHAPE instruction to acc slot (issue #192) | 2026-07-30 17:22 |
| [#189](https://github.com/rechefe/ipu-emulator/pull/189) | mode-blindness: r_cyclic holds 512 elements with a mode-dependent wrap | 2026-07-30 13:55 |
| [#188](https://github.com/rechefe/ipu-emulator/pull/188) | mode-blindness: XMEM row addressing and fixed 8 MB allocation | 2026-07-30 13:26 |
| [#191](https://github.com/rechefe/ipu-emulator/pull/191) | Mb/177 lanes | 2026-07-30 13:13 |
| [#186](https://github.com/rechefe/ipu-emulator/pull/186) | mode-blindness: separate LANES from R_REG_SIZE | 2026-07-29 18:27 |
| [#174](https://github.com/rechefe/ipu-emulator/pull/174) | Enhance instruction set and register handling for ADDB and ADDBI oper… | 2026-07-29 18:08 |
| [#185](https://github.com/rechefe/ipu-emulator/pull/185) | mode-blindness: test coverage for verified-correct debug-mode paths | 2026-07-28 17:29 |
| [#184](https://github.com/rechefe/ipu-emulator/pull/184) | mode-blindness: honor masking and pad_mode in wide-vector debug mode | 2026-07-28 17:28 |
| [#112](https://github.com/rechefe/ipu-emulator/pull/112) | Generate SystemVerilog package for instruction format (issue #110) | 2026-07-27 19:53 |
| [#172](https://github.com/rechefe/ipu-emulator/pull/172) | fix(emu): MULT reads Ra/Rc DATA from snapshot (issue #157) | 2026-07-27 18:35 |
| [#170](https://github.com/rechefe/ipu-emulator/pull/170) | Add pad_mode option to dstructure config for masked lane fill value | 2026-07-14 03:05 |
| [#168](https://github.com/rechefe/ipu-emulator/pull/168) | Update execute_activate_quantize to read live r_acc register  | 2026-07-08 18:29 |
| [#166](https://github.com/rechefe/ipu-emulator/pull/166) | Merge ACTIVATE and AAQ into single ACTIVATE.QUANTIZE instruction | 2026-06-28 18:43 |
| [#162](https://github.com/rechefe/ipu-emulator/pull/162) | Refactor AAQ instruction to ACTIVATE.QUANTIZE with INT8 quantization | 2026-06-28 18:43 |
| [#165](https://github.com/rechefe/ipu-emulator/pull/165) | Add support for Sigmoid Linear Unit (SiLU) activation function | 2026-06-28 18:32 |
| [#161](https://github.com/rechefe/ipu-emulator/pull/161) | Make dstructure CR index mandatory for masking multiply instructions | 2026-06-27 18:31 |
| [#155](https://github.com/rechefe/ipu-emulator/pull/155) | Allow per-command CR register for dstructure in mult/acc/aaq stages (#130) | 2026-06-26 12:54 |
| [#159](https://github.com/rechefe/ipu-emulator/pull/159) | Expand ACC slot with ADD/MAX/SUB operations (fixes #158) | 2026-06-24 18:46 |
| [#156](https://github.com/rechefe/ipu-emulator/pull/156) | Update documentation to reference MULT_RES lanes instead of R_ACC  | 2026-06-24 18:38 |
| [#154](https://github.com/rechefe/ipu-emulator/pull/154) | Resolve conflicts: Unify per-slot NOP variants (was #150) | 2026-06-22 19:41 |
| [#153](https://github.com/rechefe/ipu-emulator/pull/153) | Update AGG instruction references to use MULT_RES lanes  | 2026-06-22 10:01 |
| [#149](https://github.com/rechefe/ipu-emulator/pull/149) | Remove ElementsInRow=8 and expand variants from ACC.STRIDE | 2026-06-21 18:44 |
| [#148](https://github.com/rechefe/ipu-emulator/pull/148) | Remove RESET_ACC instruction | 2026-06-21 18:35 |
| [#151](https://github.com/rechefe/ipu-emulator/pull/151) | Fix BLT/BGE sign-extension at 20-bit register width | 2026-06-21 18:34 |
| [#147](https://github.com/rechefe/ipu-emulator/pull/147) | Refactor multiply instructions: consolidate MULT.EE/VE variants into RC-based API | 2026-06-20 14:01 |
| [#146](https://github.com/rechefe/ipu-emulator/pull/146) | Add pseudo-instruction support (BGT, BLE, BZ, BNZ) and BGE real instruction | 2026-06-20 12:24 |
| [#145](https://github.com/rechefe/ipu-emulator/pull/145) | Validate LDR_CYCLIC_MULT_REG index instead of silently wrapping mod 512 | 2026-06-20 11:22 |
| [#141](https://github.com/rechefe/ipu-emulator/pull/141) | Add INC/DEC to LR slot; remove AddSubSrcB from ADD/SUB (fixes #132) | 2026-06-17 20:09 |
| [#140](https://github.com/rechefe/ipu-emulator/pull/140) | LR instruction slot documented three times | 2026-06-17 19:52 |
| [#126](https://github.com/rechefe/ipu-emulator/pull/126) | AAQ: interim direct INT8 clamp instead of >>24 truncation | 2026-06-15 13:49 |
| [#125](https://github.com/rechefe/ipu-emulator/pull/125) | docs: update documentation and tests to clarify lane masking behavior | 2026-06-13 17:12 |
| [#123](https://github.com/rechefe/ipu-emulator/pull/123) | Move aggregation (SUM/MAX) from AAQ stage to ACC slot | 2026-06-13 13:38 |
| [#121](https://github.com/rechefe/ipu-emulator/pull/121) | Refactor and enhance AGG instruction tests in ipu-emu-py | 2026-06-13 13:38 |
| [#120](https://github.com/rechefe/ipu-emulator/pull/120) | ACC Slot Redesign: AAQ Operations  | 2026-06-09 19:24 |
| [#118](https://github.com/rechefe/ipu-emulator/pull/118) | Split xmem slot into load, store, and debug pipeline slots (#113) | 2026-06-09 19:17 |
| [#106](https://github.com/rechefe/ipu-emulator/pull/106) | docs: update mask_shift documentation and add sequential shift-and-AN… | 2026-06-08 19:10 |
| [#111](https://github.com/rechefe/ipu-emulator/pull/111) | Reorder compound instruction layout (issue #107) and gitignore generated docs | 2026-06-05 07:38 |
| [#104](https://github.com/rechefe/ipu-emulator/pull/104) | Plan: RISC-V host integration (SystemRDL control regfile + Unicorn + Rust firmware) | 2026-06-01 20:24 |
| [#103](https://github.com/rechefe/ipu-emulator/pull/103) | Add FullXmemRow support to instruction set and documentation | 2026-06-01 19:57 |
| [#101](https://github.com/rechefe/ipu-emulator/pull/101) | Update documentation and code to restrict usage of CR15 as an ISA ope… | 2026-05-29 16:28 |
| [#100](https://github.com/rechefe/ipu-emulator/pull/100) | Changes in IPU configuration registers (Issue #98) | 2026-05-29 14:13 |
| [#97](https://github.com/rechefe/ipu-emulator/pull/97) | Add reciprocal and rsqrt activation functions; update related specs a… | 2026-05-29 07:05 |
| [#94](https://github.com/rechefe/ipu-emulator/pull/94) | Remove Spec and AAQ Support for leaky_relu, prelu, silu/swish Activations | 2026-05-26 19:01 |
| [#79](https://github.com/rechefe/ipu-emulator/pull/79) | Issue #77: Configure activation α on IpuState (no CR) | 2026-05-23 18:52 |
| [#92](https://github.com/rechefe/ipu-emulator/pull/92) | feat(isa): union layout solver for bitfield sharing across opcodes | 2026-05-23 16:31 |
| [#90](https://github.com/rechefe/ipu-emulator/pull/90) | docs: enhance Control Stage documentation with detailed descriptions … | 2026-05-23 14:03 |
| [#89](https://github.com/rechefe/ipu-emulator/pull/89) | feat(mult): add MULT.EE.RR multi-element execution (issue #86) | 2026-05-17 17:43 |
| [#88](https://github.com/rechefe/ipu-emulator/pull/88) | docs: update ISA section with detailed instruction references for AAQ… | 2026-05-16 15:02 |
| [#81](https://github.com/rechefe/ipu-emulator/pull/81) | Control stage spec | 2026-05-15 15:11 |
| [#85](https://github.com/rechefe/ipu-emulator/pull/85) | Add execution statistics tracking (issue #84) | 2026-05-14 10:45 |
| [#83](https://github.com/rechefe/ipu-emulator/pull/83) | LR SET: copy from CR register; remove 16-bit immediate (#82) | 2026-05-14 08:50 |
| [#80](https://github.com/rechefe/ipu-emulator/pull/80) | Standardize instruction syntax and register naming conventions | 2026-05-12 20:27 |
| [#78](https://github.com/rechefe/ipu-emulator/pull/78) | Fix LR conflict detection for INCR_MOD_POW2 instructions | 2026-05-12 20:14 |
| [#72](https://github.com/rechefe/ipu-emulator/pull/72) | Standardize upper-case instruction mnemonics and lower-case operand names | 2026-05-09 19:12 |
| [#71](https://github.com/rechefe/ipu-emulator/pull/71) | Issue #51: valid_elements operand for agg / agg.first | 2026-05-09 17:32 |
| [#67](https://github.com/rechefe/ipu-emulator/pull/67) | Fix #46: immediate multiply mask slot (0–7); shift stays LR | 2026-05-09 17:22 |
| [#70](https://github.com/rechefe/ipu-emulator/pull/70) | Fix mult.ve cyclic RC access and add 128-element padding flag (closes #44) | 2026-05-09 17:15 |
| [#68](https://github.com/rechefe/ipu-emulator/pull/68) | Fix #64: add/sub ISA — LR src_a, LR\|CR\|IMM5 src_b | 2026-05-09 17:05 |
| [#69](https://github.com/rechefe/ipu-emulator/pull/69) | Close #52, #65, #66: ISA cleanup (remove mult.ev, mult.ee R0/R1 only) and docs | 2026-05-09 16:05 |
| [#62](https://github.com/rechefe/ipu-emulator/pull/62) | AAQ spec: rename Post-Function “Edge case” column to “Corner case” | 2026-05-08 13:21 |
| [#60](https://github.com/rechefe/ipu-emulator/pull/60) | MkDocs: git revision dates, history links, giscus (issue #59) | 2026-05-06 20:38 |
| [#58](https://github.com/rechefe/ipu-emulator/pull/58) | Rename "IPU Specification" to "AAQ Stage Specification" in mkdocs nav | 2026-05-06 20:04 |
| [#56](https://github.com/rechefe/ipu-emulator/pull/56) | Fix MkDocs build: nested docs paths, stage spec filename, Mermaid fences | 2026-05-06 19:50 |
| [#54](https://github.com/rechefe/ipu-emulator/pull/54) | Specs added for cache unit and AAQ stage | 2026-05-06 19:25 |
| [#50](https://github.com/rechefe/ipu-emulator/pull/50) | Encode incr_mod_pow2 k in 4 bits (k − 1) | 2026-05-05 20:29 |
| [#49](https://github.com/rechefe/ipu-emulator/pull/49) | Add incr_mod_pow2 LR instruction (Issue #47) | 2026-05-05 20:06 |
| [#48](https://github.com/rechefe/ipu-emulator/pull/48) | Allow three LR sub-instructions per VLIW word (issue #45) | 2026-05-04 19:52 |
| [#43](https://github.com/rechefe/ipu-emulator/pull/43) | Wide-vector debug mode (32-bit lanes, issue #33) | 2026-05-03 20:11 |
| [#42](https://github.com/rechefe/ipu-emulator/pull/42) | Add agg.first instruction to fix RF register noise (fixes #32) | 2026-05-03 17:31 |
| [#41](https://github.com/rechefe/ipu-emulator/pull/41) | Implements shared fixed_idx (0–255) for mult.ve | 2026-05-02 18:45 |
| [#40](https://github.com/rechefe/ipu-emulator/pull/40) | Support CR registers as operands in conditional branch and br instructions | 2026-05-02 16:12 |
| [#37](https://github.com/rechefe/ipu-emulator/pull/37) | fix: mult.ve boundary padding parity with mult.ve.cr / mult.ve.aaq | 2026-04-27 19:20 |
| [#35](https://github.com/rechefe/ipu-emulator/pull/35) | Add auto-updating Release Log documentation page | 2026-04-26 18:36 |
| [#30](https://github.com/rechefe/ipu-emulator/pull/30) | Replace bitwise stride decoding with lookup tables | 2026-04-03 08:07 |
| [#28](https://github.com/rechefe/ipu-emulator/pull/28) | Generalize FP8 support to all e(x)m(8-x) format - closes #19 | 2026-03-27 13:29 |
| [#26](https://github.com/rechefe/ipu-emulator/pull/26) | Fix duplicated legend in SVG generator | 2026-03-24 20:34 |
| [#27](https://github.com/rechefe/ipu-emulator/pull/27) | Add aaq quantization instruction and xmem.store_aaq_result (closes #24) | 2026-03-24 20:31 |
| [#23](https://github.com/rechefe/ipu-emulator/pull/23) | Add mult.ve.cr and mult.ve.aaq instructions (closes #17) | 2026-03-23 20:30 |
| [#22](https://github.com/rechefe/ipu-emulator/pull/22) | [skill] Add AI Assistant Skill Reference for IPU Emulator & Assembler | 2026-03-23 19:50 |
| [#16](https://github.com/rechefe/ipu-emulator/pull/16) | Refactor new instructions | 2026-03-07 16:05 |
| [#15](https://github.com/rechefe/ipu-emulator/pull/15) | [doc] Update docs - detailed example of fully connected layer impleme… | 2026-03-02 18:18 |
| [#14](https://github.com/rechefe/ipu-emulator/pull/14) | Python refactor | 2026-02-16 16:51 |
| [#13](https://github.com/rechefe/ipu-emulator/pull/13) | Feature/debug | 2026-01-30 15:40 |
| [#12](https://github.com/rechefe/ipu-emulator/pull/12) | [emulator] Implemented add and sub instructions for LR and CR registe… | 2026-01-30 10:57 |
| [#11](https://github.com/rechefe/ipu-emulator/pull/11) | [emulator] Fixed multiplier instruction ev and ve | 2026-01-30 10:32 |
| [#10](https://github.com/rechefe/ipu-emulator/pull/10) | Doc/adding mult acc | 2026-01-26 19:49 |
| [#8](https://github.com/rechefe/ipu-emulator/pull/8) | [docs] Add new documentat regarding how to add new instructions to as… | 2026-01-12 15:05 |
| [#7](https://github.com/rechefe/ipu-emulator/pull/7) | Reset command | 2026-01-09 14:47 |
| [#5](https://github.com/rechefe/ipu-emulator/pull/5) | [data_formats] WIP - start adding new data formats | 2026-01-09 14:47 |
| [#6](https://github.com/rechefe/ipu-emulator/pull/6) | Add zero_rq instruction and update opcode handling | 2025-12-30 19:35 |
| [#4](https://github.com/rechefe/ipu-emulator/pull/4) | Docs | 2025-12-22 20:55 |
| [#3](https://github.com/rechefe/ipu-emulator/pull/3) | Docs | 2025-12-22 20:12 |
| [#2](https://github.com/rechefe/ipu-emulator/pull/2) | [preprocessor] Add preprocessing capabilities to assembly syntax | 2025-12-22 16:29 |
| [#1](https://github.com/rechefe/ipu-emulator/pull/1) | Dockerfile | 2025-12-22 16:01 |

*Last updated: 2026-10-24 11:57 UTC*
