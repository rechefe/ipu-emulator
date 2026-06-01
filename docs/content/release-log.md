# Release Log

This page lists every pull request that has been merged into `master`, newest
first.  It is regenerated automatically as part of the docs build on every
merge.

| # | Title | Merged at (UTC) |
|---|-------|----------------|
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
| [#5](https://github.com/rechefe/ipu-emulator/pull/5) | [data_formats] WIP - start adding new data formats | 2026-01-09 14:47 |
| [#7](https://github.com/rechefe/ipu-emulator/pull/7) | Reset command | 2026-01-09 14:47 |
| [#6](https://github.com/rechefe/ipu-emulator/pull/6) | Add zero_rq instruction and update opcode handling | 2025-12-30 19:35 |
| [#4](https://github.com/rechefe/ipu-emulator/pull/4) | Docs | 2025-12-22 20:55 |
| [#3](https://github.com/rechefe/ipu-emulator/pull/3) | Docs | 2025-12-22 20:12 |
| [#2](https://github.com/rechefe/ipu-emulator/pull/2) | [preprocessor] Add preprocessing capabilities to assembly syntax | 2025-12-22 16:29 |
| [#1](https://github.com/rechefe/ipu-emulator/pull/1) | Dockerfile | 2025-12-22 16:01 |

*Last updated: 2026-05-29 18:54 UTC*
