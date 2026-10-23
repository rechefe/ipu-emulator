# IPU Assembly Instruction Reference

Per-opcode documentation is generated from `InstructionDoc` entries in `instruction_spec.py`. Operand **types** link to the shared [operand type reference](operand-types.md).

## Slots

A VLIW instruction word encodes one sub-instruction per slot. Slots are grouped into pipeline stages that execute sequentially within a cycle: **CTRL** (COND + LR run concurrently; LOAD address is resolved here and the data feeds MULT) → **MULT** → **ACC** → **AAQ** → **STORE**. `AAQ_EXACT`, `BREAK` are emulator-only: not in the hardware word. Any omitted slot is filled with `NOP` by the assembler automatically.

| Slot | Count | Tag | Description |
|------|------:|-----|-------------|
| `COND` | 1 | `HARDWARE` | Conditional and unconditional branches; control flow based on register comparisons. |
| `LR` | 3 | `HARDWARE` | Loop register instructions for controlling addresses, counters, and scalars. Three independent sub-slots per cycle. |
| `LOAD` | 1 | `HARDWARE` | First-stage memory loads that feed the multiply unit (mult-stage, cyclic, and mask registers). |
| `MULT` | 1 | `HARDWARE` | Multiply-stage instructions: element-wise and vector multiply operations. |
| `ACC` | 1 | `HARDWARE` | Accumulation instructions for combining multiply results into R_ACC. |
| `AAQ` | 1 | `HARDWARE` | Activation and quantization: activate R_ACC, quantize to one NX8 row and store it with its metadata; load the LUT segments the generic activation reads. |
| `AAQ_EXACT` | 1 | `EMULATOR` | Emulator-only exact activation: activate R_ACC with the exact Python function and store one FP32 row (FP32 mode only). Not in the hardware word. |
| `BREAK` | 1 | `EMULATOR` | Debug break: halts execution or enters debug mode. |

### `NOP` — No Operation

**Syntax:** `NOP`

**Tag:** `HARDWARE` in the hardware slots, `EMULATOR` in the emulator-only slots.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

No operation. Every slot accepts `NOP`. The assembler fills any omitted slot with `NOP` automatically. When written explicitly in a compound instruction, `NOP` is assigned to the next unfilled slot in compound-layout order (COND → LR → LOAD → MULT → ACC → AAQ → STORE → ACC_STORE → BREAK).

## Element formats

The element format is selected at run time and sets the element width. [Element formats](../../explanation/element-formats.md) lists the formats and how each encodes values.

## Compound Instruction Layout

The compound (VLIW) instruction word is shown two ways below, first as the hardware word, then as the emulator word.

* **Whole-word bit layout** — every operand token in its actual bit position (MSB → LSB), coloured by the owning slot. This is what an encoded VLIW word looks like in memory.
* **Per-slot union layout** — for each slot, the union fields packed by the layout solver, with a per-opcode grid showing which operand each opcode places in each field. Useful for seeing where cross-opcode field sharing comes from.

### Hardware word (169 bits)

The part of the instruction word the IPU decodes: the hardware slots
(`COND`, `LR`×3, `LOAD`, `MULT`, `ACC`, `AAQ`). The SystemVerilog package describes this
word (`IPU_COMPOUND_INST_WIDTH = 169`).

#### Whole-word bit layout

<svg width="888" height="706" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .inst-title { font-size: 14px; font-weight: bold; font-family: Arial; }
      .inst-label { font-size: 8px; font-weight: bold; font-family: Arial; }
      .field-label { font-size: 7px; font-family: Arial; }
      .bit-range { font-size: 8px; font-family: Arial; }
    </style>
  </defs>
  <rect x="0" y="0" width="888" height="706" fill="white" stroke="none"/>
  <text x="444.0" y="34" text-anchor="middle" class="inst-title">Hardware word Layout - 169 total bits</text>
  <rect x="80" y="55.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="99.0" text-anchor="end" class="bit-range">[168:160]</text>
  <text x="644.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">168</text>
  <text x="668.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">167</text>
  <text x="692.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">166</text>
  <text x="716.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">165</text>
  <text x="740.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">164</text>
  <text x="764.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">163</text>
  <text x="788.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">162</text>
  <text x="812.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">161</text>
  <text x="836.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">160</text>
  <rect x="704" y="55.0" width="144" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="776.0" y="84.0" text-anchor="middle" class="field-label">label</text>
  <text x="776.0" y="95.0" text-anchor="middle" class="field-label">token</text>
  <text x="776.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[165:156]</text>
  <rect x="632" y="55.0" width="72" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="668.0" y="78.5" text-anchor="middle" class="field-label">cond</text>
  <text x="668.0" y="89.5" text-anchor="middle" class="field-label">inst</text>
  <text x="668.0" y="100.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="668.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[168:166]</text>
  <rect x="80" y="135.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="179.0" text-anchor="end" class="bit-range">[159:128]</text>
  <text x="92.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">159</text>
  <text x="116.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">158</text>
  <text x="140.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">157</text>
  <text x="164.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">156</text>
  <text x="188.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">155</text>
  <text x="212.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">154</text>
  <text x="236.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">153</text>
  <text x="260.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">152</text>
  <text x="284.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">151</text>
  <text x="308.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">150</text>
  <text x="332.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">149</text>
  <text x="356.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">148</text>
  <text x="380.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">147</text>
  <text x="404.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">146</text>
  <text x="428.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">145</text>
  <text x="452.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">144</text>
  <text x="476.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">143</text>
  <text x="500.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">142</text>
  <text x="524.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">141</text>
  <text x="548.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">140</text>
  <text x="572.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">139</text>
  <text x="596.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">138</text>
  <text x="620.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">137</text>
  <text x="644.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">136</text>
  <text x="668.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">135</text>
  <text x="692.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">134</text>
  <text x="716.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">133</text>
  <text x="740.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">132</text>
  <text x="764.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">131</text>
  <text x="788.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">130</text>
  <text x="812.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">129</text>
  <text x="836.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">128</text>
  <rect x="752" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="800.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[131:128]</text>
  <rect x="560" y="135.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="656.0" y="164.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="656.0" y="175.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="656.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[139:132]</text>
  <rect x="464" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="512.0" y="158.5" text-anchor="middle" class="field-label">lr</text>
  <text x="512.0" y="169.5" text-anchor="middle" class="field-label">inst</text>
  <text x="512.0" y="180.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="512.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[143:140]</text>
  <rect x="320" y="135.0" width="144" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="392.0" y="158.5" text-anchor="middle" class="field-label">lcr</text>
  <text x="392.0" y="169.5" text-anchor="middle" class="field-label">half</text>
  <text x="392.0" y="180.5" text-anchor="middle" class="field-label">reg</text>
  <text x="392.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[149:144]</text>
  <rect x="176" y="135.0" width="144" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="248.0" y="158.5" text-anchor="middle" class="field-label">lcr</text>
  <text x="248.0" y="169.5" text-anchor="middle" class="field-label">half</text>
  <text x="248.0" y="180.5" text-anchor="middle" class="field-label">reg</text>
  <text x="248.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[155:150]</text>
  <rect x="80" y="135.0" width="96" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="128.0" y="164.0" text-anchor="middle" class="field-label">label</text>
  <text x="128.0" y="175.0" text-anchor="middle" class="field-label">token</text>
  <text x="128.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[165:156]</text>
  <rect x="80" y="215.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="259.0" text-anchor="end" class="bit-range">[127:96]</text>
  <text x="92.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">127</text>
  <text x="116.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">126</text>
  <text x="140.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">125</text>
  <text x="164.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">124</text>
  <text x="188.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">123</text>
  <text x="212.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">122</text>
  <text x="236.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">121</text>
  <text x="260.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">120</text>
  <text x="284.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">119</text>
  <text x="308.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">118</text>
  <text x="332.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">117</text>
  <text x="356.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">116</text>
  <text x="380.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">115</text>
  <text x="404.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">114</text>
  <text x="428.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">113</text>
  <text x="452.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">112</text>
  <text x="476.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">111</text>
  <text x="500.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">110</text>
  <text x="524.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">109</text>
  <text x="548.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">108</text>
  <text x="572.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">107</text>
  <text x="596.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">106</text>
  <text x="620.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">105</text>
  <text x="644.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">104</text>
  <text x="668.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">103</text>
  <text x="692.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">102</text>
  <text x="716.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">101</text>
  <text x="740.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">100</text>
  <text x="764.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">99</text>
  <text x="788.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">98</text>
  <text x="812.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">97</text>
  <text x="836.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">96</text>
  <rect x="752" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="244.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="800.0" y="255.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="800.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[99:92]</text>
  <rect x="656" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="704.0" y="238.5" text-anchor="middle" class="field-label">lr</text>
  <text x="704.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="704.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="704.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[103:100]</text>
  <rect x="560" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="608.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="608.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="608.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[107:104]</text>
  <rect x="464" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="512.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="512.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="512.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[111:108]</text>
  <rect x="272" y="215.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="368.0" y="244.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="368.0" y="255.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="368.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[119:112]</text>
  <rect x="176" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="224.0" y="238.5" text-anchor="middle" class="field-label">lr</text>
  <text x="224.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="224.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="224.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[123:120]</text>
  <rect x="80" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="128.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="128.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="128.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[127:124]</text>
  <rect x="80" y="295.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="339.0" text-anchor="end" class="bit-range">[95:64]</text>
  <text x="92.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">95</text>
  <text x="116.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">94</text>
  <text x="140.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">93</text>
  <text x="164.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">92</text>
  <text x="188.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">91</text>
  <text x="212.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">90</text>
  <text x="236.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">89</text>
  <text x="260.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">88</text>
  <text x="284.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">87</text>
  <text x="308.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">86</text>
  <text x="332.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">85</text>
  <text x="356.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">84</text>
  <text x="380.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">83</text>
  <text x="404.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">82</text>
  <text x="428.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">81</text>
  <text x="452.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">80</text>
  <text x="476.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">79</text>
  <text x="500.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">78</text>
  <text x="524.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">77</text>
  <text x="548.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">76</text>
  <text x="572.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">75</text>
  <text x="596.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">74</text>
  <text x="620.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">73</text>
  <text x="644.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">72</text>
  <text x="668.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">71</text>
  <text x="692.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">70</text>
  <text x="716.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">69</text>
  <text x="740.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">68</text>
  <text x="764.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">67</text>
  <text x="788.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">66</text>
  <text x="812.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">65</text>
  <text x="836.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">64</text>
  <rect x="776" y="295.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="812.0" y="324.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="812.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="812.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[66:62]</text>
  <rect x="704" y="295.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="740.0" y="318.5" text-anchor="middle" class="field-label">mult</text>
  <text x="740.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="740.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="740.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[69:67]</text>
  <rect x="608" y="295.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="656.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="656.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="656.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[73:70]</text>
  <rect x="512" y="295.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="560.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="560.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="560.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[77:74]</text>
  <rect x="416" y="295.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="464.0" y="324.0" text-anchor="middle" class="field-label">cr</text>
  <text x="464.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="464.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[81:78]</text>
  <rect x="368" y="295.0" width="48" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="392.0" y="318.5" text-anchor="middle" class="field-label">load</text>
  <text x="392.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="392.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="392.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[83:82]</text>
  <rect x="272" y="295.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="320.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="320.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="320.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[87:84]</text>
  <rect x="176" y="295.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="224.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="224.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="224.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[91:88]</text>
  <rect x="80" y="295.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="128.0" y="324.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="128.0" y="335.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="128.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[99:92]</text>
  <rect x="80" y="375.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="419.0" text-anchor="end" class="bit-range">[63:32]</text>
  <text x="92.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">63</text>
  <text x="116.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">62</text>
  <text x="140.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">61</text>
  <text x="164.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">60</text>
  <text x="188.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">59</text>
  <text x="212.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">58</text>
  <text x="236.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">57</text>
  <text x="260.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">56</text>
  <text x="284.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">55</text>
  <text x="308.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">54</text>
  <text x="332.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">53</text>
  <text x="356.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">52</text>
  <text x="380.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">51</text>
  <text x="404.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">50</text>
  <text x="428.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">49</text>
  <text x="452.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">48</text>
  <text x="476.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">47</text>
  <text x="500.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">46</text>
  <text x="524.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">45</text>
  <text x="548.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">44</text>
  <text x="572.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">43</text>
  <text x="596.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">42</text>
  <text x="620.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">41</text>
  <text x="644.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">40</text>
  <text x="668.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">39</text>
  <text x="692.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">38</text>
  <text x="716.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">37</text>
  <text x="740.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">36</text>
  <text x="764.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">35</text>
  <text x="788.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">34</text>
  <text x="812.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">33</text>
  <text x="836.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">32</text>
  <rect x="728" y="375.0" width="120" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="788.0" y="398.5" text-anchor="middle" class="field-label">half</text>
  <text x="788.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="788.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="788.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[36:32]</text>
  <rect x="632" y="375.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="680.0" y="398.5" text-anchor="middle" class="field-label">acc</text>
  <text x="680.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="680.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="680.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[40:37]</text>
  <rect x="560" y="375.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="596.0" y="393.0" text-anchor="middle" class="field-label">mult</text>
  <text x="596.0" y="404.0" text-anchor="middle" class="field-label">mask</text>
  <text x="596.0" y="415.0" text-anchor="middle" class="field-label">offset</text>
  <text x="596.0" y="426.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="596.0" y="440.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[43:41]</text>
  <rect x="464" y="375.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="512.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="512.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="512.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[47:44]</text>
  <rect x="368" y="375.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="416.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="416.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="416.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[51:48]</text>
  <rect x="248" y="375.0" width="120" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="308.0" y="398.5" text-anchor="middle" class="field-label">half</text>
  <text x="308.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="308.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="308.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[56:52]</text>
  <rect x="128" y="375.0" width="120" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="188.0" y="398.5" text-anchor="middle" class="field-label">half</text>
  <text x="188.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="188.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="188.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[61:57]</text>
  <rect x="80" y="375.0" width="48" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="104.0" y="404.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="104.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="104.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[66:62]</text>
  <rect x="80" y="455.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="499.0" text-anchor="end" class="bit-range">[31:0]</text>
  <text x="92.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">31</text>
  <text x="116.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">30</text>
  <text x="140.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">29</text>
  <text x="164.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">28</text>
  <text x="188.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">27</text>
  <text x="212.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">26</text>
  <text x="236.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">25</text>
  <text x="260.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">24</text>
  <text x="284.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">23</text>
  <text x="308.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">22</text>
  <text x="332.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">21</text>
  <text x="356.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">20</text>
  <text x="380.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">19</text>
  <text x="404.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">18</text>
  <text x="428.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">17</text>
  <text x="452.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">16</text>
  <text x="476.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">15</text>
  <text x="500.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">14</text>
  <text x="524.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">13</text>
  <text x="548.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">12</text>
  <text x="572.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">11</text>
  <text x="596.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">10</text>
  <text x="620.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">9</text>
  <text x="644.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">8</text>
  <text x="668.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">7</text>
  <text x="692.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">6</text>
  <text x="716.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">5</text>
  <text x="740.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">4</text>
  <text x="764.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">3</text>
  <text x="788.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">2</text>
  <text x="812.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">1</text>
  <text x="836.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">0</text>
  <rect x="752" y="455.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="484.0" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="800.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[3:0]</text>
  <rect x="632" y="455.0" width="120" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="692.0" y="478.5" text-anchor="middle" class="field-label">half</text>
  <text x="692.0" y="489.5" text-anchor="middle" class="field-label">cr</text>
  <text x="692.0" y="500.5" text-anchor="middle" class="field-label">reg</text>
  <text x="692.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[8:4]</text>
  <rect x="512" y="455.0" width="120" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="572.0" y="478.5" text-anchor="middle" class="field-label">half</text>
  <text x="572.0" y="489.5" text-anchor="middle" class="field-label">cr</text>
  <text x="572.0" y="500.5" text-anchor="middle" class="field-label">reg</text>
  <text x="572.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[13:9]</text>
  <rect x="416" y="455.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="464.0" y="484.0" text-anchor="middle" class="field-label">cr</text>
  <text x="464.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="464.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[17:14]</text>
  <rect x="344" y="455.0" width="72" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="380.0" y="478.5" text-anchor="middle" class="field-label">aaq</text>
  <text x="380.0" y="489.5" text-anchor="middle" class="field-label">activation</text>
  <text x="380.0" y="500.5" text-anchor="middle" class="field-label">fn</text>
  <text x="380.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[20:18]</text>
  <rect x="296" y="455.0" width="48" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="320.0" y="478.5" text-anchor="middle" class="field-label">aaq</text>
  <text x="320.0" y="489.5" text-anchor="middle" class="field-label">inst</text>
  <text x="320.0" y="500.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="320.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[22:21]</text>
  <rect x="248" y="455.0" width="48" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="272.0" y="484.0" text-anchor="middle" class="field-label">vertical</text>
  <text x="272.0" y="495.0" text-anchor="middle" class="field-label">stride</text>
  <text x="272.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[24:23]</text>
  <rect x="152" y="455.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="200.0" y="484.0" text-anchor="middle" class="field-label">lr</text>
  <text x="200.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="200.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[28:25]</text>
  <rect x="80" y="455.0" width="72" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="116.0" y="484.0" text-anchor="middle" class="field-label">lrd</text>
  <text x="116.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="116.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[31:29]</text>
  <text x="80" y="555.0" class="inst-label" style="font-weight: bold;">Instruction Type Colors:</text>
  <rect x="80" y="560.0" width="12" height="12" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="572.0" class="field-label" style="font-size: 9px;">LoadInst (Memory Load)</text>
  <rect x="80" y="575.0" width="12" height="12" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="587.0" class="field-label" style="font-size: 9px;">MultInst (Multiply)</text>
  <rect x="80" y="590.0" width="12" height="12" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="602.0" class="field-label" style="font-size: 9px;">AccInst (Accumulator)</text>
  <rect x="80" y="605.0" width="12" height="12" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="617.0" class="field-label" style="font-size: 9px;">AaqInst (Activation and Quantization)</text>
  <rect x="80" y="620.0" width="12" height="12" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="632.0" class="field-label" style="font-size: 9px;">LrInst (Link Register)</text>
  <rect x="80" y="635.0" width="12" height="12" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="647.0" class="field-label" style="font-size: 9px;">CondInst (Conditional)</text>
</svg>

#### Per-slot union layout

<svg xmlns="http://www.w3.org/2000/svg" width="660" height="1456" viewBox="0 0 660 1456">
  <rect x="0" y="0" width="660" height="1456" fill="white"/>
  <text x="330" y="38" text-anchor="middle" font-size="14" font-weight="bold" font-family="Arial">IPU VLIW — Union Field Layout (per slot, MSB → LSB)</text>
  <text x="330" y="68" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">COND  (25 bits = 3b opcode + 10+6+6b fields)</text>
  <rect x="24" y="76" width="73" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="60" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="60" y="118" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <rect x="97" y="76" width="244" height="64" fill="#98D8C8" stroke="#333" stroke-width="1" rx="3"/>
  <text x="219" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">Label</text>
  <text x="219" y="118" text-anchor="middle" font-size="8" font-family="Arial">10b</text>
  <rect x="341" y="76" width="146" height="64" fill="#98D8C8" stroke="#333" stroke-width="1" rx="3"/>
  <text x="414" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LcrHalfIdx</text>
  <text x="414" y="118" text-anchor="middle" font-size="8" font-family="Arial">6b</text>
  <rect x="487" y="76" width="149" height="64" fill="#98D8C8" stroke="#333" stroke-width="1" rx="3"/>
  <text x="561" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LcrHalfIdx</text>
  <text x="561" y="118" text-anchor="middle" font-size="8" font-family="Arial">6b</text>
  <text x="28" y="156" font-size="8.5" font-family="monospace" fill="#222">BEQ</text>
  <rect x="134" y="147" width="200" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">label</text>
  <rect x="334" y="147" width="119" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="393" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">reg1</text>
  <rect x="453" y="147" width="183" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="544" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">reg2</text>
  <text x="28" y="170" font-size="8.5" font-family="monospace" fill="#222">BGE.S</text>
  <rect x="134" y="161" width="200" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">label</text>
  <rect x="334" y="161" width="119" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="393" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">reg1</text>
  <rect x="453" y="161" width="183" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="544" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">reg2</text>
  <text x="28" y="184" font-size="8.5" font-family="monospace" fill="#222">BLT.S</text>
  <rect x="134" y="175" width="200" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="184" text-anchor="middle" font-size="7.5" font-family="Arial">label</text>
  <rect x="334" y="175" width="119" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="393" y="184" text-anchor="middle" font-size="7.5" font-family="Arial">reg1</text>
  <rect x="453" y="175" width="183" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="544" y="184" text-anchor="middle" font-size="7.5" font-family="Arial">reg2</text>
  <text x="28" y="198" font-size="8.5" font-family="monospace" fill="#222">BNE</text>
  <rect x="134" y="189" width="200" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="198" text-anchor="middle" font-size="7.5" font-family="Arial">label</text>
  <rect x="334" y="189" width="119" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="393" y="198" text-anchor="middle" font-size="7.5" font-family="Arial">reg1</text>
  <rect x="453" y="189" width="183" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="544" y="198" text-anchor="middle" font-size="7.5" font-family="Arial">reg2</text>
  <text x="28" y="212" font-size="8.5" font-family="monospace" fill="#222">BR</text>
  <rect x="134" y="203" width="200" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="212" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="334" y="203" width="119" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="393" y="212" text-anchor="middle" font-size="7.5" font-family="Arial">reg</text>
  <rect x="453" y="203" width="183" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="544" y="212" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="226" font-size="8.5" font-family="monospace" fill="#222">HALT</text>
  <rect x="134" y="217" width="200" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="226" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="334" y="217" width="119" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="393" y="226" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="453" y="217" width="183" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="544" y="226" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="240" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="231" width="200" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="240" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="334" y="231" width="119" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="393" y="240" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="453" y="231" width="183" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="544" y="240" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="330" y="298" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">LR (×3)  (20 bits = 4b opcode + 8+4+4b fields)</text>
  <rect x="24" y="306" width="122" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="85" y="334" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="85" y="348" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="146" y="306" width="244" height="64" fill="#FFA07A" stroke="#333" stroke-width="1" rx="3"/>
  <text x="268" y="334" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">AddbiImmediate</text>
  <text x="268" y="348" text-anchor="middle" font-size="8" font-family="Arial">8b</text>
  <rect x="390" y="306" width="122" height="64" fill="#FFA07A" stroke="#333" stroke-width="1" rx="3"/>
  <text x="451" y="334" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="451" y="348" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="512" y="306" width="124" height="64" fill="#FFA07A" stroke="#333" stroke-width="1" rx="3"/>
  <text x="574" y="334" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="574" y="348" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <text x="28" y="386" font-size="8.5" font-family="monospace" fill="#222">ADD</text>
  <rect x="134" y="377" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="386" text-anchor="middle" font-size="7.5" font-family="Arial">src_b:LcrHalfIdx</text>
  <rect x="334" y="377" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="386" text-anchor="middle" font-size="7.5" font-family="Arial">dest</text>
  <rect x="434" y="377" width="202" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="386" text-anchor="middle" font-size="7.5" font-family="Arial">src_a</text>
  <text x="28" y="400" font-size="8.5" font-family="monospace" fill="#222">ADDB</text>
  <rect x="134" y="391" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="400" text-anchor="middle" font-size="7.5" font-family="Arial">src_b:LcrHalfIdx</text>
  <rect x="334" y="391" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="400" text-anchor="middle" font-size="7.5" font-family="Arial">dest:LrdIdx</text>
  <rect x="434" y="391" width="202" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="400" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="414" font-size="8.5" font-family="monospace" fill="#222">ADDBI</text>
  <rect x="134" y="405" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="414" text-anchor="middle" font-size="7.5" font-family="Arial">imm</text>
  <rect x="334" y="405" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="414" text-anchor="middle" font-size="7.5" font-family="Arial">dest:LrdIdx</text>
  <rect x="434" y="405" width="202" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="414" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="428" font-size="8.5" font-family="monospace" fill="#222">DEC</text>
  <rect x="134" y="419" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="428" text-anchor="middle" font-size="7.5" font-family="Arial">imm:LrIncDecImmediate</text>
  <rect x="334" y="419" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="428" text-anchor="middle" font-size="7.5" font-family="Arial">dest</text>
  <rect x="434" y="419" width="202" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="428" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="442" font-size="8.5" font-family="monospace" fill="#222">INC</text>
  <rect x="134" y="433" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="442" text-anchor="middle" font-size="7.5" font-family="Arial">imm:LrIncDecImmediate</text>
  <rect x="334" y="433" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="442" text-anchor="middle" font-size="7.5" font-family="Arial">dest</text>
  <rect x="434" y="433" width="202" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="442" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="456" font-size="8.5" font-family="monospace" fill="#222">INC.MODP2</text>
  <rect x="134" y="447" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="456" text-anchor="middle" font-size="7.5" font-family="Arial">step:LcrHalfIdx</text>
  <rect x="334" y="447" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="456" text-anchor="middle" font-size="7.5" font-family="Arial">dest</text>
  <rect x="434" y="447" width="202" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="456" text-anchor="middle" font-size="7.5" font-family="Arial">k:LrModPow2KImmediate</text>
  <text x="28" y="470" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="461" width="200" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="470" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="334" y="461" width="100" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="470" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="434" y="461" width="202" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="470" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="484" font-size="8.5" font-family="monospace" fill="#222">SET</text>
  <rect x="134" y="475" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="484" text-anchor="middle" font-size="7.5" font-family="Arial">src:LcrHalfIdx</text>
  <rect x="334" y="475" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="484" text-anchor="middle" font-size="7.5" font-family="Arial">reg</text>
  <rect x="434" y="475" width="202" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="484" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="498" font-size="8.5" font-family="monospace" fill="#222">SUB</text>
  <rect x="134" y="489" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="498" text-anchor="middle" font-size="7.5" font-family="Arial">src_b:LcrHalfIdx</text>
  <rect x="334" y="489" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="498" text-anchor="middle" font-size="7.5" font-family="Arial">dest</text>
  <rect x="434" y="489" width="202" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="498" text-anchor="middle" font-size="7.5" font-family="Arial">src_a</text>
  <text x="330" y="556" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">LOAD  (14 bits = 2b opcode + 4+4+4b fields)</text>
  <rect x="24" y="564" width="87" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="67" y="592" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="67" y="606" text-anchor="middle" font-size="8" font-family="Arial">2b</text>
  <rect x="111" y="564" width="174" height="64" fill="#FF6B6B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="198" y="592" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">CrIdx</text>
  <text x="198" y="606" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="285" y="564" width="174" height="64" fill="#FF6B6B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="372" y="592" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="372" y="606" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="459" y="564" width="177" height="64" fill="#FF6B6B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="547" y="592" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="547" y="606" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <text x="28" y="644" font-size="8.5" font-family="monospace" fill="#222">LD.MASK</text>
  <rect x="134" y="635" width="142" height="12" fill="#FF6B6B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="205" y="644" text-anchor="middle" font-size="7.5" font-family="Arial">base</text>
  <rect x="276" y="635" width="142" height="12" fill="#FF6B6B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="347" y="644" text-anchor="middle" font-size="7.5" font-family="Arial">offset</text>
  <rect x="418" y="635" width="218" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="527" y="644" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="658" font-size="8.5" font-family="monospace" fill="#222">LD.MULT</text>
  <rect x="134" y="649" width="142" height="12" fill="#FF6B6B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="205" y="658" text-anchor="middle" font-size="7.5" font-family="Arial">base</text>
  <rect x="276" y="649" width="142" height="12" fill="#FF6B6B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="347" y="658" text-anchor="middle" font-size="7.5" font-family="Arial">offset</text>
  <rect x="418" y="649" width="218" height="12" fill="#FF6B6B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="527" y="658" text-anchor="middle" font-size="7.5" font-family="Arial">dest:MultStageReg</text>
  <text x="28" y="672" font-size="8.5" font-family="monospace" fill="#222">LD.MULT.CYCLIC</text>
  <rect x="134" y="663" width="142" height="12" fill="#FF6B6B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="205" y="672" text-anchor="middle" font-size="7.5" font-family="Arial">base</text>
  <rect x="276" y="663" width="142" height="12" fill="#FF6B6B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="347" y="672" text-anchor="middle" font-size="7.5" font-family="Arial">offset</text>
  <rect x="418" y="663" width="218" height="12" fill="#FF6B6B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="527" y="672" text-anchor="middle" font-size="7.5" font-family="Arial">index</text>
  <text x="28" y="686" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="677" width="142" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="205" y="686" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="276" y="677" width="142" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="347" y="686" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="418" y="677" width="218" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="527" y="686" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="330" y="744" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">MULT  (29 bits = 3b opcode + 5+5+5+4+4+3b fields)</text>
  <rect x="24" y="752" width="63" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="55" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="55" y="794" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <rect x="87" y="752" width="105" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="139" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LcrIdx</text>
  <text x="139" y="794" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="192" y="752" width="105" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="244" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">HalfCrIdx</text>
  <text x="244" y="794" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="297" y="752" width="105" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="349" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">HalfCrIdx</text>
  <text x="349" y="794" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="402" y="752" width="84" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="444" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="444" y="794" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="486" y="752" width="84" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="528" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="528" y="794" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="570" y="752" width="66" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="603" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">MultMaskOffsetImmediate</text>
  <text x="603" y="794" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <text x="28" y="832" font-size="8.5" font-family="monospace" fill="#222">MULT.EE</text>
  <rect x="134" y="823" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="177" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx:CrIdx</text>
  <rect x="220" y="823" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="263" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="306" y="823" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="349" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">qf</text>
  <rect x="392" y="823" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="426" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">ra_idx</text>
  <rect x="460" y="823" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="494" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="528" y="823" width="108" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="582" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="846" font-size="8.5" font-family="monospace" fill="#222">MULT.RC.VE</text>
  <rect x="134" y="837" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="177" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">src</text>
  <rect x="220" y="837" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="263" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="306" y="837" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="349" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">qf</text>
  <rect x="392" y="837" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="426" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">rc_idx</text>
  <rect x="460" y="837" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="494" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="528" y="837" width="108" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="582" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="860" font-size="8.5" font-family="monospace" fill="#222">MULT.RC.VS</text>
  <rect x="134" y="851" width="86" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="177" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="220" y="851" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="263" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="306" y="851" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="349" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">qf</text>
  <rect x="392" y="851" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="426" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">rc_idx</text>
  <rect x="460" y="851" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="494" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="528" y="851" width="108" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="582" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="874" font-size="8.5" font-family="monospace" fill="#222">MULT.RC.VV</text>
  <rect x="134" y="865" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="177" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">ra:MultStageReg</text>
  <rect x="220" y="865" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="263" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="306" y="865" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="349" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">qf</text>
  <rect x="392" y="865" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="426" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">rc_idx</text>
  <rect x="460" y="865" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="494" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="528" y="865" width="108" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="582" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="888" font-size="8.5" font-family="monospace" fill="#222">MULT.VE</text>
  <rect x="134" y="879" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="177" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx:CrIdx</text>
  <rect x="220" y="879" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="263" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="306" y="879" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="349" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">qf</text>
  <rect x="392" y="879" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="426" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">ra_idx</text>
  <rect x="460" y="879" width="68" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="494" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="528" y="879" width="108" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="582" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="902" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="893" width="86" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="177" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="220" y="893" width="86" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="263" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="306" y="893" width="86" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="349" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="392" y="893" width="68" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="426" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="460" y="893" width="68" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="494" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="528" y="893" width="108" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="582" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="330" y="960" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">ACC  (18 bits = 4b opcode + 5+3+4+2b fields)</text>
  <rect x="24" y="968" width="136" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="92" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="92" y="1010" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="160" y="968" width="170" height="64" fill="#45B7D1" stroke="#333" stroke-width="1" rx="3"/>
  <text x="245" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">HalfCrIdx</text>
  <text x="245" y="1010" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="330" y="968" width="102" height="64" fill="#45B7D1" stroke="#333" stroke-width="1" rx="3"/>
  <text x="381" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrdIdx</text>
  <text x="381" y="1010" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <rect x="432" y="968" width="136" height="64" fill="#45B7D1" stroke="#333" stroke-width="1" rx="3"/>
  <text x="500" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="500" y="1010" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="568" y="968" width="68" height="64" fill="#45B7D1" stroke="#333" stroke-width="1" rx="3"/>
  <text x="602" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">VerticalStride</text>
  <text x="602" y="1010" text-anchor="middle" font-size="8" font-family="Arial">2b</text>
  <text x="28" y="1048" font-size="8.5" font-family="monospace" fill="#222">ACC.ADD</text>
  <rect x="134" y="1039" width="139" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1048" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="273" y="1039" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1048" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1039" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1048" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="467" y="1039" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1048" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1062" font-size="8.5" font-family="monospace" fill="#222">ACC.ADD.FIRST</text>
  <rect x="134" y="1053" width="139" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1062" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="273" y="1053" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1062" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1053" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1062" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="467" y="1053" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1062" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1076" font-size="8.5" font-family="monospace" fill="#222">ACC.MAX</text>
  <rect x="134" y="1067" width="139" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1076" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="273" y="1067" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1076" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1067" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1076" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="467" y="1067" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1076" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1090" font-size="8.5" font-family="monospace" fill="#222">ACC.MAX.FIRST</text>
  <rect x="134" y="1081" width="139" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1090" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="273" y="1081" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1090" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1081" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1090" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="467" y="1081" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1090" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1104" font-size="8.5" font-family="monospace" fill="#222">ACC.RESHAPE</text>
  <rect x="134" y="1095" width="139" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1104" text-anchor="middle" font-size="7.5" font-family="Arial">reshape_mask:LrOrReshapeMaskImmediate</text>
  <rect x="273" y="1095" width="83" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1104" text-anchor="middle" font-size="7.5" font-family="Arial">source</text>
  <rect x="356" y="1095" width="111" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1104" text-anchor="middle" font-size="7.5" font-family="Arial">dest:LrdIdx</text>
  <rect x="467" y="1095" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1104" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1118" font-size="8.5" font-family="monospace" fill="#222">ACC.STRIDE</text>
  <rect x="134" y="1109" width="139" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1118" text-anchor="middle" font-size="7.5" font-family="Arial">elements_in_row:ElementsInRow</text>
  <rect x="273" y="1109" width="83" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1118" text-anchor="middle" font-size="7.5" font-family="Arial">horizontal_stride:HorizontalStride</text>
  <rect x="356" y="1109" width="111" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1118" text-anchor="middle" font-size="7.5" font-family="Arial">offset</text>
  <rect x="467" y="1109" width="169" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1118" text-anchor="middle" font-size="7.5" font-family="Arial">vertical_stride</text>
  <text x="28" y="1132" font-size="8.5" font-family="monospace" fill="#222">ACC.SUB</text>
  <rect x="134" y="1123" width="139" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1132" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="273" y="1123" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1132" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1123" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1132" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="467" y="1123" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1132" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1146" font-size="8.5" font-family="monospace" fill="#222">ACC.SUB.FIRST</text>
  <rect x="134" y="1137" width="139" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1146" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="273" y="1137" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1146" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1137" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1146" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="467" y="1137" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1146" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1160" font-size="8.5" font-family="monospace" fill="#222">AGG.MAX</text>
  <rect x="134" y="1151" width="139" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1160" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="273" y="1151" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1160" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1151" width="111" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1160" text-anchor="middle" font-size="7.5" font-family="Arial">dest_slot</text>
  <rect x="467" y="1151" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1160" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1174" font-size="8.5" font-family="monospace" fill="#222">AGG.MAX.FIRST</text>
  <rect x="134" y="1165" width="139" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1174" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="273" y="1165" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1174" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1165" width="111" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1174" text-anchor="middle" font-size="7.5" font-family="Arial">dest_slot</text>
  <rect x="467" y="1165" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1174" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1188" font-size="8.5" font-family="monospace" fill="#222">AGG.SUM</text>
  <rect x="134" y="1179" width="139" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1188" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="273" y="1179" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1188" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1179" width="111" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1188" text-anchor="middle" font-size="7.5" font-family="Arial">dest_slot</text>
  <rect x="467" y="1179" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1188" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1202" font-size="8.5" font-family="monospace" fill="#222">AGG.SUM.FIRST</text>
  <rect x="134" y="1193" width="139" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1202" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="273" y="1193" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1202" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1193" width="111" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1202" text-anchor="middle" font-size="7.5" font-family="Arial">dest_slot</text>
  <rect x="467" y="1193" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1202" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1216" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="1207" width="139" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="203" y="1216" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="273" y="1207" width="83" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="314" y="1216" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="1207" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="411" y="1216" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="467" y="1207" width="169" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="551" y="1216" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="330" y="1274" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">AAQ  (23 bits = 2b opcode + 3+4+5+5+4b fields)</text>
  <rect x="24" y="1282" width="53" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="50" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="50" y="1324" text-anchor="middle" font-size="8" font-family="Arial">2b</text>
  <rect x="77" y="1282" width="79" height="64" fill="#9B59B6" stroke="#333" stroke-width="1" rx="3"/>
  <text x="116" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">AaqActivationFn</text>
  <text x="116" y="1324" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <rect x="156" y="1282" width="106" height="64" fill="#9B59B6" stroke="#333" stroke-width="1" rx="3"/>
  <text x="209" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">CrIdx</text>
  <text x="209" y="1324" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="262" y="1282" width="133" height="64" fill="#9B59B6" stroke="#333" stroke-width="1" rx="3"/>
  <text x="328" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">HalfCrIdx</text>
  <text x="328" y="1324" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="395" y="1282" width="133" height="64" fill="#9B59B6" stroke="#333" stroke-width="1" rx="3"/>
  <text x="461" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">HalfCrIdx</text>
  <text x="461" y="1324" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="528" y="1282" width="108" height="64" fill="#9B59B6" stroke="#333" stroke-width="1" rx="3"/>
  <text x="582" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="582" y="1324" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <text x="28" y="1362" font-size="8.5" font-family="monospace" fill="#222">ACTIVATE.QUANTIZE</text>
  <rect x="134" y="1353" width="64" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="166" y="1362" text-anchor="middle" font-size="7.5" font-family="Arial">activation_fn</text>
  <rect x="198" y="1353" width="86" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="241" y="1362" text-anchor="middle" font-size="7.5" font-family="Arial">table_id</text>
  <rect x="284" y="1353" width="109" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="338" y="1362" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="393" y="1353" width="109" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="447" y="1362" text-anchor="middle" font-size="7.5" font-family="Arial">qf</text>
  <rect x="502" y="1353" width="134" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="569" y="1362" text-anchor="middle" font-size="7.5" font-family="Arial">table_offset</text>
  <text x="28" y="1376" font-size="8.5" font-family="monospace" fill="#222">LD.LUT</text>
  <rect x="134" y="1367" width="64" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="166" y="1376" text-anchor="middle" font-size="7.5" font-family="Arial">lut_addr:LutAddrImmediate</text>
  <rect x="198" y="1367" width="86" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="241" y="1376" text-anchor="middle" font-size="7.5" font-family="Arial">table_id</text>
  <rect x="284" y="1367" width="109" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="338" y="1376" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="393" y="1367" width="109" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="447" y="1376" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="502" y="1367" width="134" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="569" y="1376" text-anchor="middle" font-size="7.5" font-family="Arial">table_offset</text>
  <text x="28" y="1390" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="1381" width="64" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="166" y="1390" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="198" y="1381" width="86" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="241" y="1390" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="284" y="1381" width="109" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="338" y="1390" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="393" y="1381" width="109" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="447" y="1390" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="502" y="1381" width="134" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="569" y="1390" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
</svg>

### Emulator word (209 bits)

The word the emulator decodes and `.bin` files hold: the hardware word followed by
the emulator-only slots (`AAQ_EXACT`, `BREAK`) below its bit 0. Its top
169 bits are the hardware word.

#### Emulator-only slots: union layout

<svg xmlns="http://www.w3.org/2000/svg" width="660" height="410" viewBox="0 0 660 410">
  <rect x="0" y="0" width="660" height="410" fill="white"/>
  <text x="330" y="38" text-anchor="middle" font-size="14" font-weight="bold" font-family="Arial">IPU VLIW — Union Field Layout (per slot, MSB → LSB)</text>
  <text x="330" y="68" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">AAQ_EXACT  (18 bits = 1b opcode + 4+4+5+4b fields)</text>
  <rect x="24" y="76" width="34" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="41" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="41" y="118" text-anchor="middle" font-size="8" font-family="Arial">1b</text>
  <rect x="58" y="76" width="136" height="64" fill="#C0392B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="126" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">ActivationFn</text>
  <text x="126" y="118" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="194" y="76" width="136" height="64" fill="#C0392B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="262" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">CrIdx</text>
  <text x="262" y="118" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="330" y="76" width="170" height="64" fill="#C0392B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="415" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">HalfCrIdx</text>
  <text x="415" y="118" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="500" y="76" width="136" height="64" fill="#C0392B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="568" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="568" y="118" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <text x="28" y="156" font-size="8.5" font-family="monospace" fill="#222">ACTIVATE.EXACT</text>
  <rect x="134" y="147" width="111" height="12" fill="#C0392B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="189" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">activation_fn</text>
  <rect x="245" y="147" width="111" height="12" fill="#C0392B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="300" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">table_id</text>
  <rect x="356" y="147" width="139" height="12" fill="#C0392B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="425" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">ds</text>
  <rect x="495" y="147" width="141" height="12" fill="#C0392B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="565" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">table_offset</text>
  <text x="28" y="170" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="161" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="189" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="245" y="161" width="111" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="300" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="356" y="161" width="139" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="425" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="495" y="161" width="141" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="565" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="330" y="228" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">BREAK  (22 bits = 2b opcode + 16+4b fields)</text>
  <rect x="24" y="236" width="55" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="51" y="264" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="51" y="278" text-anchor="middle" font-size="8" font-family="Arial">2b</text>
  <rect x="79" y="236" width="445" height="64" fill="#FFD93D" stroke="#333" stroke-width="1" rx="3"/>
  <text x="301" y="264" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">BreakImmediate</text>
  <text x="301" y="278" text-anchor="middle" font-size="8" font-family="Arial">16b</text>
  <rect x="524" y="236" width="112" height="64" fill="#FFD93D" stroke="#333" stroke-width="1" rx="3"/>
  <text x="580" y="264" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="580" y="278" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <text x="28" y="316" font-size="8.5" font-family="monospace" fill="#222">BREAK</text>
  <rect x="134" y="307" width="365" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="316" y="316" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="499" y="307" width="137" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="567" y="316" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="330" font-size="8.5" font-family="monospace" fill="#222">BREAK.IFEQ</text>
  <rect x="134" y="321" width="365" height="12" fill="#FFD93D55" stroke="#bbb" stroke-width="0.5"/>
  <text x="316" y="330" text-anchor="middle" font-size="7.5" font-family="Arial">value</text>
  <rect x="499" y="321" width="137" height="12" fill="#FFD93D55" stroke="#bbb" stroke-width="0.5"/>
  <text x="567" y="330" text-anchor="middle" font-size="7.5" font-family="Arial">reg</text>
  <text x="28" y="344" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="335" width="365" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="316" y="344" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="499" y="335" width="137" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="567" y="344" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
</svg>

#### Whole-word bit layout

<svg width="888" height="816" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .inst-title { font-size: 14px; font-weight: bold; font-family: Arial; }
      .inst-label { font-size: 8px; font-weight: bold; font-family: Arial; }
      .field-label { font-size: 7px; font-family: Arial; }
      .bit-range { font-size: 8px; font-family: Arial; }
    </style>
  </defs>
  <rect x="0" y="0" width="888" height="816" fill="white" stroke="none"/>
  <text x="444.0" y="34" text-anchor="middle" class="inst-title">Emulator word Layout - 209 total bits</text>
  <rect x="80" y="55.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="99.0" text-anchor="end" class="bit-range">[208:192]</text>
  <text x="452.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">208</text>
  <text x="476.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">207</text>
  <text x="500.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">206</text>
  <text x="524.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">205</text>
  <text x="548.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">204</text>
  <text x="572.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">203</text>
  <text x="596.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">202</text>
  <text x="620.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">201</text>
  <text x="644.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">200</text>
  <text x="668.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">199</text>
  <text x="692.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">198</text>
  <text x="716.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">197</text>
  <text x="740.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">196</text>
  <text x="764.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">195</text>
  <text x="788.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">194</text>
  <text x="812.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">193</text>
  <text x="836.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">192</text>
  <rect x="752" y="55.0" width="96" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="78.5" text-anchor="middle" class="field-label">lcr</text>
  <text x="800.0" y="89.5" text-anchor="middle" class="field-label">half</text>
  <text x="800.0" y="100.5" text-anchor="middle" class="field-label">reg</text>
  <text x="800.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[195:190]</text>
  <rect x="512" y="55.0" width="240" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="632.0" y="84.0" text-anchor="middle" class="field-label">label</text>
  <text x="632.0" y="95.0" text-anchor="middle" class="field-label">token</text>
  <text x="632.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[205:196]</text>
  <rect x="440" y="55.0" width="72" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="476.0" y="78.5" text-anchor="middle" class="field-label">cond</text>
  <text x="476.0" y="89.5" text-anchor="middle" class="field-label">inst</text>
  <text x="476.0" y="100.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="476.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[208:206]</text>
  <rect x="80" y="135.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="179.0" text-anchor="end" class="bit-range">[191:160]</text>
  <text x="92.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">191</text>
  <text x="116.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">190</text>
  <text x="140.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">189</text>
  <text x="164.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">188</text>
  <text x="188.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">187</text>
  <text x="212.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">186</text>
  <text x="236.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">185</text>
  <text x="260.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">184</text>
  <text x="284.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">183</text>
  <text x="308.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">182</text>
  <text x="332.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">181</text>
  <text x="356.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">180</text>
  <text x="380.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">179</text>
  <text x="404.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">178</text>
  <text x="428.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">177</text>
  <text x="452.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">176</text>
  <text x="476.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">175</text>
  <text x="500.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">174</text>
  <text x="524.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">173</text>
  <text x="548.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">172</text>
  <text x="572.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">171</text>
  <text x="596.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">170</text>
  <text x="620.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">169</text>
  <text x="644.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">168</text>
  <text x="668.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">167</text>
  <text x="692.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">166</text>
  <text x="716.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">165</text>
  <text x="740.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">164</text>
  <text x="764.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">163</text>
  <text x="788.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">162</text>
  <text x="812.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">161</text>
  <text x="836.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">160</text>
  <rect x="752" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="158.5" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="169.5" text-anchor="middle" class="field-label">inst</text>
  <text x="800.0" y="180.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="800.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[163:160]</text>
  <rect x="656" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="704.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="704.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="704.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[167:164]</text>
  <rect x="560" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="608.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="608.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="608.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[171:168]</text>
  <rect x="368" y="135.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="464.0" y="164.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="464.0" y="175.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="464.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[179:172]</text>
  <rect x="272" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="320.0" y="158.5" text-anchor="middle" class="field-label">lr</text>
  <text x="320.0" y="169.5" text-anchor="middle" class="field-label">inst</text>
  <text x="320.0" y="180.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="320.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[183:180]</text>
  <rect x="128" y="135.0" width="144" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="200.0" y="158.5" text-anchor="middle" class="field-label">lcr</text>
  <text x="200.0" y="169.5" text-anchor="middle" class="field-label">half</text>
  <text x="200.0" y="180.5" text-anchor="middle" class="field-label">reg</text>
  <text x="200.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[189:184]</text>
  <rect x="80" y="135.0" width="48" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="104.0" y="158.5" text-anchor="middle" class="field-label">lcr</text>
  <text x="104.0" y="169.5" text-anchor="middle" class="field-label">half</text>
  <text x="104.0" y="180.5" text-anchor="middle" class="field-label">reg</text>
  <text x="104.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[195:190]</text>
  <rect x="80" y="215.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="259.0" text-anchor="end" class="bit-range">[159:128]</text>
  <text x="92.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">159</text>
  <text x="116.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">158</text>
  <text x="140.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">157</text>
  <text x="164.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">156</text>
  <text x="188.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">155</text>
  <text x="212.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">154</text>
  <text x="236.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">153</text>
  <text x="260.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">152</text>
  <text x="284.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">151</text>
  <text x="308.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">150</text>
  <text x="332.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">149</text>
  <text x="356.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">148</text>
  <text x="380.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">147</text>
  <text x="404.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">146</text>
  <text x="428.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">145</text>
  <text x="452.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">144</text>
  <text x="476.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">143</text>
  <text x="500.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">142</text>
  <text x="524.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">141</text>
  <text x="548.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">140</text>
  <text x="572.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">139</text>
  <text x="596.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">138</text>
  <text x="620.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">137</text>
  <text x="644.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">136</text>
  <text x="668.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">135</text>
  <text x="692.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">134</text>
  <text x="716.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">133</text>
  <text x="740.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">132</text>
  <text x="764.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">131</text>
  <text x="788.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">130</text>
  <text x="812.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">129</text>
  <text x="836.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">128</text>
  <rect x="752" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="800.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[131:128]</text>
  <rect x="560" y="215.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="656.0" y="244.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="656.0" y="255.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="656.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[139:132]</text>
  <rect x="464" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="512.0" y="238.5" text-anchor="middle" class="field-label">lr</text>
  <text x="512.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="512.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="512.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[143:140]</text>
  <rect x="368" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="416.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="416.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="416.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[147:144]</text>
  <rect x="272" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="320.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="320.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="320.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[151:148]</text>
  <rect x="80" y="215.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="176.0" y="244.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="176.0" y="255.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="176.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[159:152]</text>
  <rect x="80" y="295.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="339.0" text-anchor="end" class="bit-range">[127:96]</text>
  <text x="92.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">127</text>
  <text x="116.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">126</text>
  <text x="140.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">125</text>
  <text x="164.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">124</text>
  <text x="188.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">123</text>
  <text x="212.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">122</text>
  <text x="236.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">121</text>
  <text x="260.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">120</text>
  <text x="284.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">119</text>
  <text x="308.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">118</text>
  <text x="332.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">117</text>
  <text x="356.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">116</text>
  <text x="380.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">115</text>
  <text x="404.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">114</text>
  <text x="428.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">113</text>
  <text x="452.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">112</text>
  <text x="476.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">111</text>
  <text x="500.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">110</text>
  <text x="524.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">109</text>
  <text x="548.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">108</text>
  <text x="572.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">107</text>
  <text x="596.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">106</text>
  <text x="620.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">105</text>
  <text x="644.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">104</text>
  <text x="668.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">103</text>
  <text x="692.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">102</text>
  <text x="716.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">101</text>
  <text x="740.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">100</text>
  <text x="764.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">99</text>
  <text x="788.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">98</text>
  <text x="812.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">97</text>
  <text x="836.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">96</text>
  <rect x="824" y="295.0" width="24" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="836.0" y="318.5" text-anchor="middle" class="field-label">half</text>
  <text x="836.0" y="329.5" text-anchor="middle" class="field-label">cr</text>
  <text x="836.0" y="340.5" text-anchor="middle" class="field-label">reg</text>
  <text x="836.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[96:92]</text>
  <rect x="704" y="295.0" width="120" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="764.0" y="318.5" text-anchor="middle" class="field-label">half</text>
  <text x="764.0" y="329.5" text-anchor="middle" class="field-label">cr</text>
  <text x="764.0" y="340.5" text-anchor="middle" class="field-label">reg</text>
  <text x="764.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[101:97]</text>
  <rect x="584" y="295.0" width="120" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="644.0" y="324.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="644.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="644.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[106:102]</text>
  <rect x="512" y="295.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="548.0" y="318.5" text-anchor="middle" class="field-label">mult</text>
  <text x="548.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="548.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="548.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[109:107]</text>
  <rect x="416" y="295.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="464.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="464.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="464.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[113:110]</text>
  <rect x="320" y="295.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="368.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="368.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="368.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[117:114]</text>
  <rect x="224" y="295.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="272.0" y="324.0" text-anchor="middle" class="field-label">cr</text>
  <text x="272.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="272.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[121:118]</text>
  <rect x="176" y="295.0" width="48" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="200.0" y="318.5" text-anchor="middle" class="field-label">load</text>
  <text x="200.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="200.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="200.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[123:122]</text>
  <rect x="80" y="295.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="128.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="128.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="128.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[127:124]</text>
  <rect x="80" y="375.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="419.0" text-anchor="end" class="bit-range">[95:64]</text>
  <text x="92.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">95</text>
  <text x="116.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">94</text>
  <text x="140.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">93</text>
  <text x="164.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">92</text>
  <text x="188.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">91</text>
  <text x="212.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">90</text>
  <text x="236.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">89</text>
  <text x="260.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">88</text>
  <text x="284.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">87</text>
  <text x="308.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">86</text>
  <text x="332.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">85</text>
  <text x="356.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">84</text>
  <text x="380.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">83</text>
  <text x="404.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">82</text>
  <text x="428.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">81</text>
  <text x="452.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">80</text>
  <text x="476.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">79</text>
  <text x="500.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">78</text>
  <text x="524.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">77</text>
  <text x="548.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">76</text>
  <text x="572.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">75</text>
  <text x="596.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">74</text>
  <text x="620.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">73</text>
  <text x="644.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">72</text>
  <text x="668.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">71</text>
  <text x="692.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">70</text>
  <text x="716.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">69</text>
  <text x="740.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">68</text>
  <text x="764.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">67</text>
  <text x="788.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">66</text>
  <text x="812.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">65</text>
  <text x="836.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">64</text>
  <rect x="824" y="375.0" width="24" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="836.0" y="404.0" text-anchor="middle" class="field-label">vertical</text>
  <text x="836.0" y="415.0" text-anchor="middle" class="field-label">stride</text>
  <text x="836.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[64:63]</text>
  <rect x="728" y="375.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="776.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="776.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="776.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[68:65]</text>
  <rect x="656" y="375.0" width="72" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="692.0" y="404.0" text-anchor="middle" class="field-label">lrd</text>
  <text x="692.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="692.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[71:69]</text>
  <rect x="536" y="375.0" width="120" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="596.0" y="398.5" text-anchor="middle" class="field-label">half</text>
  <text x="596.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="596.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="596.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[76:72]</text>
  <rect x="440" y="375.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="488.0" y="398.5" text-anchor="middle" class="field-label">acc</text>
  <text x="488.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="488.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="488.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[80:77]</text>
  <rect x="368" y="375.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="404.0" y="393.0" text-anchor="middle" class="field-label">mult</text>
  <text x="404.0" y="404.0" text-anchor="middle" class="field-label">mask</text>
  <text x="404.0" y="415.0" text-anchor="middle" class="field-label">offset</text>
  <text x="404.0" y="426.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="404.0" y="440.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[83:81]</text>
  <rect x="272" y="375.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="320.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="320.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="320.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[87:84]</text>
  <rect x="176" y="375.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="224.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="224.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="224.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[91:88]</text>
  <rect x="80" y="375.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="128.0" y="398.5" text-anchor="middle" class="field-label">half</text>
  <text x="128.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="128.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="128.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[96:92]</text>
  <rect x="80" y="455.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="499.0" text-anchor="end" class="bit-range">[63:32]</text>
  <text x="92.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">63</text>
  <text x="116.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">62</text>
  <text x="140.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">61</text>
  <text x="164.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">60</text>
  <text x="188.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">59</text>
  <text x="212.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">58</text>
  <text x="236.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">57</text>
  <text x="260.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">56</text>
  <text x="284.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">55</text>
  <text x="308.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">54</text>
  <text x="332.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">53</text>
  <text x="356.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">52</text>
  <text x="380.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">51</text>
  <text x="404.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">50</text>
  <text x="428.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">49</text>
  <text x="452.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">48</text>
  <text x="476.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">47</text>
  <text x="500.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">46</text>
  <text x="524.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">45</text>
  <text x="548.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">44</text>
  <text x="572.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">43</text>
  <text x="596.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">42</text>
  <text x="620.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">41</text>
  <text x="644.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">40</text>
  <text x="668.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">39</text>
  <text x="692.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">38</text>
  <text x="716.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">37</text>
  <text x="740.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">36</text>
  <text x="764.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">35</text>
  <text x="788.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">34</text>
  <text x="812.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">33</text>
  <text x="836.0" y="467.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">32</text>
  <rect x="776" y="455.0" width="72" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="812.0" y="484.0" text-anchor="middle" class="field-label">cr</text>
  <text x="812.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="812.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[34:31]</text>
  <rect x="680" y="455.0" width="96" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="728.0" y="484.0" text-anchor="middle" class="field-label">activation</text>
  <text x="728.0" y="495.0" text-anchor="middle" class="field-label">fn</text>
  <text x="728.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[38:35]</text>
  <rect x="656" y="455.0" width="24" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="668.0" y="473.0" text-anchor="middle" class="field-label">aaq</text>
  <text x="668.0" y="484.0" text-anchor="middle" class="field-label">exact</text>
  <text x="668.0" y="495.0" text-anchor="middle" class="field-label">inst</text>
  <text x="668.0" y="506.0" text-anchor="middle" class="field-label">opcode</text>
  <text x="668.0" y="520.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[39:39]</text>
  <rect x="560" y="455.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="608.0" y="484.0" text-anchor="middle" class="field-label">lr</text>
  <text x="608.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="608.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[43:40]</text>
  <rect x="440" y="455.0" width="120" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="500.0" y="478.5" text-anchor="middle" class="field-label">half</text>
  <text x="500.0" y="489.5" text-anchor="middle" class="field-label">cr</text>
  <text x="500.0" y="500.5" text-anchor="middle" class="field-label">reg</text>
  <text x="500.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[48:44]</text>
  <rect x="320" y="455.0" width="120" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="380.0" y="478.5" text-anchor="middle" class="field-label">half</text>
  <text x="380.0" y="489.5" text-anchor="middle" class="field-label">cr</text>
  <text x="380.0" y="500.5" text-anchor="middle" class="field-label">reg</text>
  <text x="380.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[53:49]</text>
  <rect x="224" y="455.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="272.0" y="484.0" text-anchor="middle" class="field-label">cr</text>
  <text x="272.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="272.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[57:54]</text>
  <rect x="152" y="455.0" width="72" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="188.0" y="478.5" text-anchor="middle" class="field-label">aaq</text>
  <text x="188.0" y="489.5" text-anchor="middle" class="field-label">activation</text>
  <text x="188.0" y="500.5" text-anchor="middle" class="field-label">fn</text>
  <text x="188.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[60:58]</text>
  <rect x="104" y="455.0" width="48" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="128.0" y="478.5" text-anchor="middle" class="field-label">aaq</text>
  <text x="128.0" y="489.5" text-anchor="middle" class="field-label">inst</text>
  <text x="128.0" y="500.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="128.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[62:61]</text>
  <rect x="80" y="455.0" width="24" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="484.0" text-anchor="middle" class="field-label">vertical</text>
  <text x="92.0" y="495.0" text-anchor="middle" class="field-label">stride</text>
  <text x="92.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[64:63]</text>
  <rect x="80" y="535.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="579.0" text-anchor="end" class="bit-range">[31:0]</text>
  <text x="92.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">31</text>
  <text x="116.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">30</text>
  <text x="140.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">29</text>
  <text x="164.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">28</text>
  <text x="188.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">27</text>
  <text x="212.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">26</text>
  <text x="236.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">25</text>
  <text x="260.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">24</text>
  <text x="284.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">23</text>
  <text x="308.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">22</text>
  <text x="332.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">21</text>
  <text x="356.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">20</text>
  <text x="380.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">19</text>
  <text x="404.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">18</text>
  <text x="428.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">17</text>
  <text x="452.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">16</text>
  <text x="476.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">15</text>
  <text x="500.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">14</text>
  <text x="524.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">13</text>
  <text x="548.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">12</text>
  <text x="572.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">11</text>
  <text x="596.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">10</text>
  <text x="620.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">9</text>
  <text x="644.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">8</text>
  <text x="668.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">7</text>
  <text x="692.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">6</text>
  <text x="716.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">5</text>
  <text x="740.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">4</text>
  <text x="764.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">3</text>
  <text x="788.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">2</text>
  <text x="812.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">1</text>
  <text x="836.0" y="547.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">0</text>
  <rect x="752" y="535.0" width="96" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="564.0" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="575.0" text-anchor="middle" class="field-label">reg</text>
  <text x="800.0" y="589.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[3:0]</text>
  <rect x="368" y="535.0" width="384" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="560.0" y="564.0" text-anchor="middle" class="field-label">break</text>
  <text x="560.0" y="575.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="560.0" y="589.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[19:4]</text>
  <rect x="320" y="535.0" width="48" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="344.0" y="558.5" text-anchor="middle" class="field-label">break</text>
  <text x="344.0" y="569.5" text-anchor="middle" class="field-label">inst</text>
  <text x="344.0" y="580.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="344.0" y="594.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[21:20]</text>
  <rect x="224" y="535.0" width="96" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="272.0" y="564.0" text-anchor="middle" class="field-label">lr</text>
  <text x="272.0" y="575.0" text-anchor="middle" class="field-label">reg</text>
  <text x="272.0" y="589.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[25:22]</text>
  <rect x="104" y="535.0" width="120" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="164.0" y="558.5" text-anchor="middle" class="field-label">half</text>
  <text x="164.0" y="569.5" text-anchor="middle" class="field-label">cr</text>
  <text x="164.0" y="580.5" text-anchor="middle" class="field-label">reg</text>
  <text x="164.0" y="594.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[30:26]</text>
  <rect x="80" y="535.0" width="24" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="564.0" text-anchor="middle" class="field-label">cr</text>
  <text x="92.0" y="575.0" text-anchor="middle" class="field-label">reg</text>
  <text x="92.0" y="589.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[34:31]</text>
  <text x="80" y="635.0" class="inst-label" style="font-weight: bold;">Instruction Type Colors:</text>
  <rect x="80" y="640.0" width="12" height="12" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="652.0" class="field-label" style="font-size: 9px;">BreakInst (Break / Debug)</text>
  <rect x="80" y="655.0" width="12" height="12" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="667.0" class="field-label" style="font-size: 9px;">LoadInst (Memory Load)</text>
  <rect x="80" y="670.0" width="12" height="12" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="682.0" class="field-label" style="font-size: 9px;">MultInst (Multiply)</text>
  <rect x="80" y="685.0" width="12" height="12" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="697.0" class="field-label" style="font-size: 9px;">AccInst (Accumulator)</text>
  <rect x="80" y="700.0" width="12" height="12" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="712.0" class="field-label" style="font-size: 9px;">AaqInst (Activation and Quantization)</text>
  <rect x="80" y="715.0" width="12" height="12" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="727.0" class="field-label" style="font-size: 9px;">AaqExactInst (ACTIVATE.EXACT, emulator-only)</text>
  <rect x="80" y="730.0" width="12" height="12" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="742.0" class="field-label" style="font-size: 9px;">LrInst (Link Register)</text>
  <rect x="80" y="745.0" width="12" height="12" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="757.0" class="field-label" style="font-size: 9px;">CondInst (Conditional)</text>
</svg>


---


## Conditional Branch Instructions

Control flow instructions for branching based on conditions or unconditionally.

### `BEQ` — Branch if Equal

**Syntax:** `BEQ reg1 reg2 label`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg1: First register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `reg2` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg2: Second register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if two registers are equal.

**Pseudo code:**
`if (reg1 == reg2) PC = label`

**Example of usage:**
```asm
BEQ LR0 LR1 end;;
end:
HALT;;
```

**Register reads:** `reg1` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg1` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `reg2` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg2` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `BNE` — Branch if Not Equal

**Syntax:** `BNE reg1 reg2 label`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg1: First register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `reg2` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg2: Second register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if two registers are not equal.

**Pseudo code:**
`if (reg1 != reg2) PC = label`

**Example of usage:**
```asm
loop:
INC LR0 1;;
BNE LR0 CR0 loop;;
```

**Register reads:** `reg1` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg1` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `reg2` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg2` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `BLT.S` — Branch if Less Than (signed)

**Syntax:** `BLT.S reg1 reg2 label`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg1: First register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `reg2` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg2: Second register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if first register is less than second, as signed 32-bit integers.

**Pseudo code:**
`if (int32(reg1) < int32(reg2)) PC = label  # signed 32-bit`

**Example of usage:**
```asm
BLT.S LR0 CR1 smaller;;
smaller:
HALT;;
```

**Register reads:** `reg1` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg1` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `reg2` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg2` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `BGE.S` — Branch if Greater or Equal (signed)

**Syntax:** `BGE.S reg1 reg2 label`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg1: First register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `reg2` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg2: Second register to compare (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if first register is greater than or equal to second, as signed 32-bit integers.

**Pseudo code:**
`if (int32(reg1) >= int32(reg2)) PC = label  # signed 32-bit`

**Example of usage:**
```asm
BGE.S LR0 CR1 ge;;
ge:
HALT;;
```

**Register reads:** `reg1` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg1` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `reg2` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg2` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `BR` — Branch Register

**Syntax:** `BR reg`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | reg: Register containing target address (LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H) |

**General description:**
Branch to address in register.

**Pseudo code:**
`PC = reg`


**Register reads:** `reg` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1); `reg` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `HALT` — Halt

**Syntax:** `HALT`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
End the kernel after every slot of this word has retired.

> Unconditional. The other slots of the same word retire first; then the kernel is done. To stop for debugging, use **`BREAK`** (emulator-only) instead. Not the host's `HALT` command, which stops a running kernel.

**Pseudo code:**
`no further fetch; kernel done when this word retires`

**Example of usage:**
```asm
HALT;;
```

---

## LR Instructions

Loop register manipulation instructions for controlling loop counters and addresses.

### `SET` — Set Loop Register

**Syntax:** `SET reg src`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LrIdx`](operand-types.md#lridx) | reg: Loop register (LR0–LR15) |
| `src` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | src: Source register (LR0–LR15, CR0–CR15, or a half CR0.L–CR15.H, sign-extended) |

**General description:**
Copy a 32-bit value (an LR, a CR, or a sign-extended CR half) into a loop register.

**Pseudo code:**
`reg = src  # a half is sign-extended from 16 bits`

**Example of usage:**
```asm
SET LR0 CR1;;
SET LR1 CR5.H;;
```

**Register reads:** `src` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4); `src` (`CR`): [cycle start](../../explanation/timing-model.md#rule-4).

### `ADD` — Add

**Syntax:** `ADD dest src_a src_b`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination local register (LR0–LR15) |
| `src_a` | [`LrIdx`](operand-types.md#lridx) | src_a: First source local register (LR0–LR15) |
| `src_b` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | src_b: Second source — LR0–LR15, CR0–CR15, or a half CR0.L–CR15.H (sign-extended) |

**General description:**
Add two register sources and store the result in the destination LR.

**Pseudo code:**
`dest = src_a + src_b`

**Example of usage:**
```asm
ADD LR0 LR1 LR2;;
ADD LR3 LR1 CR5;;
```

**Register reads:** `src_a` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4); `src_b` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4); `src_b` (`CR`): [cycle start](../../explanation/timing-model.md#rule-4).

### `SUB` — Subtract

**Syntax:** `SUB dest src_a src_b`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination local register (LR0–LR15) |
| `src_a` | [`LrIdx`](operand-types.md#lridx) | src_a: First source local register (LR0–LR15) |
| `src_b` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | src_b: Second source — LR0–LR15, CR0–CR15, or a half CR0.L–CR15.H (sign-extended) |

**General description:**
Subtract the second source from the first and store the result in the destination LR.

**Pseudo code:**
`dest = src_a - src_b`

**Example of usage:**
```asm
SUB LR0 LR1 LR2;;
SUB LR3 LR1 CR5;;
```

**Register reads:** `src_a` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4); `src_b` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4); `src_b` (`CR`): [cycle start](../../explanation/timing-model.md#rule-4).

### `INC.MODP2` — Increment Loop Register Modulo Power of Two

**Syntax:** `INC.MODP2 dest step k`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-1, F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination loop register (LR0–LR15); read and written |
| `step` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | step: Signed 32-bit increment from LR0–LR15, CR0–CR15, or a sign-extended half CR0.L–CR15.H |
| `k` | [`LrModPow2KImmediate`](operand-types.md#lrmodpow2kimmediate) | k: Immediate in [1, 9]; encoded in 4 bits as (k − 1); mask = (1 << k) - 1 |

**General description:**
Add a loop or configuration register into the destination loop register, then mask to k low bits (mod 2^k).

**Pseudo code:**
`dest <- (dest + step) & ((1 << k) - 1)`

**Example of usage:**
```asm
INC.MODP2 LR2 LR3 4;;
```

**Register reads:** `step` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4); `step` (`CR`): [cycle start](../../explanation/timing-model.md#rule-4); `dest` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4).

### `INC` — Increment

**Syntax:** `INC dest imm`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination local register (LR0–LR15); also the implicit source |
| `imm` | [`LrIncDecImmediate`](operand-types.md#lrincdecimmediate) | imm: Unsigned immediate; range 0 to 2^W − 1 where W is derived from the LR slot union layout |

**General description:**
Add an unsigned immediate to the destination LR (read-modify-write).

**Pseudo code:**
`dest = dest + imm`

**Example of usage:**
```asm
INC LR0 7;;
```

**Register reads:** `dest` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4).

### `DEC` — Decrement

**Syntax:** `DEC dest imm`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination local register (LR0–LR15); also the implicit source |
| `imm` | [`LrIncDecImmediate`](operand-types.md#lrincdecimmediate) | imm: Unsigned immediate; range 0 to 2^W − 1 where W is derived from the LR slot union layout |

**General description:**
Subtract an unsigned immediate from the destination LR (read-modify-write).

**Pseudo code:**
`dest = dest - imm`

**Example of usage:**
```asm
DEC LR0 3;;
```

**Register reads:** `dest` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4).

### `ADDB` — Add Byte (Broadcast, Register)

**Syntax:** `ADDB dest src_b`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrdIdx`](operand-types.md#lrdidx) | dest: Destination register pair (LRD0, LRD2, LRD4, ..., LRD14 — named after the lower register); LRDn = LR(n+1):LR(n), 8 byte elements; also the implicit source |
| `src_b` | [`LcrHalfIdx`](operand-types.md#lcrhalfidx) | src_b: LR0–LR15, CR0–CR15, or a half CR0.L–CR15.H; low byte is read and reinterpreted as a signed two's-complement byte |

**General description:**
Broadcast-add a signed byte from an LR/CR register's low byte to each of the 8 byte elements of an LRDn register pair, saturating each element to [0, 255].

> There is no separate subtract opcode: since src_b's byte is reinterpreted as signed, a value ≥ 128 (i.e. negative in two's complement) subtracts from every element instead of adding. E.g. an LR/CR whose low byte is 200 (== -56) decreases each element by 56. Because elements saturate at both ends of [0, 255], subtracting past 0 clamps at 0 rather than wrapping — it does not behave like a plain two's-complement subtract.

**Pseudo code:**
`byte_val = src_b & 0xFF  # interpreted as signed int8 (two's complement)
for i in 0..7: dest_byte[i] = clamp(dest_byte[i] + byte_val, 0, 255)`

**Example of usage:**
```asm
ADDB LRD0 LR3;;
ADDB LRD2 CR5;;
```

**Register reads:** `src_b` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4); `src_b` (`CR`): [cycle start](../../explanation/timing-model.md#rule-4); `dest` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4).

### `ADDBI` — Add Byte Immediate (Broadcast)

**Syntax:** `ADDBI dest imm`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrdIdx`](operand-types.md#lrdidx) | dest: Destination register pair (LRD0, LRD2, LRD4, ..., LRD14 — named after the lower register); LRDn = LR(n+1):LR(n), 8 byte elements; also the implicit source |
| `imm` | [`AddbiImmediate`](operand-types.md#addbiimmediate) | imm: Unsigned byte 0–255 (or an equivalent signed −128–127 literal); reinterpreted as a signed two's-complement byte for the add |

**General description:**
Broadcast-add a signed immediate byte to each of the 8 byte elements of an LRDn register pair, saturating each element to [0, 255].

> There is no separate subtract opcode: write imm as a negative literal (e.g. -56) or its equivalent unsigned encoding (200) to subtract from every element instead of adding — both spellings encode the same bit pattern. Because elements saturate at both ends of [0, 255], subtracting past 0 clamps at 0 rather than wrapping — it does not behave like a plain two's-complement subtract.

**Pseudo code:**
`byte_val = imm  # interpreted as signed int8 (two's complement)
for i in 0..7: dest_byte[i] = clamp(dest_byte[i] + byte_val, 0, 255)`

**Example of usage:**
```asm
ADDBI LRD0 200;;
ADDBI LRD2 7;;
```

**Register reads:** `dest` (`LR`): [cycle start](../../explanation/timing-model.md#rule-4).

---

## LOAD Instructions

First-stage memory loads that feed the multiply unit (mult-stage registers, cyclic register, and mask register).

### `LD.MULT` — Load Register

**Syntax:** `LD.MULT dest offset base`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-1, F-6, F-7 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`MultStageReg`](operand-types.md#multstagereg) | `dest`: **`R0`** \| **`R1`** — mult-stage register to load (2-bit field; only these encodings are valid). |
| `offset` | [`LrIdx`](operand-types.md#lridx) | `offset`: **`LR0`**…**`LR15`** — offset register, in XMEM rows. |
| `base` | [`CrIdx`](operand-types.md#cridx) | `base`: **`CR0`**…**`CR15`** — base address register, in XMEM rows. |

**General description:**
Load data from memory into a multiplication stage register.

**Pseudo code:**
`dest = Memory[row(offset + base)]  # one row = 128 elements: 128 B in NX8 mode, 512 B in FP32 mode
dest.metadata = Metadata[row(offset + base)]  # NX8: scale, dynamic_exponent; FP32: 0`

**Example of usage:**
```asm
SET LR0 CR1;;
LD.MULT R0 LR0 CR0;;
```

**Register reads:** `offset` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `base` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `LD.MULT.CYCLIC` — Load Cyclic Register

**Syntax:** `LD.MULT.CYCLIC offset base index`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-4, F-6, F-7 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15), in XMEM rows. |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR15), in XMEM rows. |
| `index` | [`LrIdx`](operand-types.md#lridx) | index: Index inside cyclic register (LR0–LR15); must hold 0, 128, 256, or 384 — the four R_CYCLIC slot boundaries, counted in **elements**. Any other value raises an error. All four are valid in **both** modes: R_CYCLIC is a 512-element ring in NX8 and FP32 mode alike, and a load always replaces one whole 128-element slot and its metadata. |

**General description:**
Load with cyclic addressing into R_CYCLIC.

**Pseudo code:**
`R_CYCLIC[index .. index+127] = Memory[row(offset + base)]   # index ∈ {0, 128, 256, 384}
R_CYCLIC.metadata[index / 128] = Metadata[row(offset + base)]  # NX8; 0 in FP32 mode`


**Register reads:** `offset` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `base` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `index` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1).

### `LD.MASK` — Load Mask Register

**Syntax:** `LD.MASK offset base`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-6, F-7 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15), in XMEM rows. |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR15), in XMEM rows. |

**General description:**
Load mask data from memory.

**Pseudo code:**
`R_MASK = Memory[row(offset + base) .. +128B]  # reads only the row's first 128 B, in both modes -- the mask is 1 bit/element and does not scale with element width`


**Register reads:** `offset` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `base` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

---

## MULT Instructions

Multiplication instructions for element-wise and element-vector operations.
The multiplication result (`mult_result`) is forwarded to the ACC stage in the CPU and not stored in any register in the way.

### `MULT.RC.VV` — RC Vector × Ra Vector Multiply

**Syntax:** `MULT.RC.VV rc_idx ra mask_offset mask_shift ds qf`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-1 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](operand-types.md#lridx) | `rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`). |
| `ra` | [`MultStageReg`](operand-types.md#multstagereg) | `ra`: **`R0`** \| **`R1`** — multiplicand mult-stage register (same cycle as `LD.MULT` into **`R0`**/**`R1`** is allowed). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | `ds`: **`CR0.L`**…**`CR15.H`** — DS half-CR (`ipu_isa.cr_halves`): `num_partitions` builds the partition vectors for `mask_shift`, `mask_type` is the value written into deactivated elements (0, +inf or −inf; NaN is an emulator error). The whole DS is validated. |
| `qf` | [`HalfCrIdx`](operand-types.md#halfcridx) | `qf`: **`CR0.L`**…**`CR15.H`** — QF half-CR naming the NX8 format of the vector elements: in NX8 mode each element expands to FP32 with its row's metadata (`ARCH-stage-mult.md` §3.2.1); ignored in FP32 mode. |

**General description:**
Multiply R_CYCLIC[rc_idx:rc_idx+128] by Ra (R0 or R1) element-wise. The partition used by mask_shift and the mask pad come from the DS half `ds`; in NX8 mode the elements are decoded with the QF half `qf` and their rows' metadata.

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the DS `mask_type` value (0, +inf or −inf) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`For each element i: MULT_RES[i] = expand(R_CYCLIC[(rc_idx + i) % 512]) * expand(ra[i]); then apply mask and shift using DS[ds].num_partitions and DS[ds].mask_type.`

**Example of usage:**
```asm
MULT.RC.VV LR0 R0 0 LR2 CR15.L CR14.L;;
```

**Register reads:** `rc_idx` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `ra` (`R`): [cycle start](../../explanation/timing-model.md#rule-2); `mask_shift` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `R_CYCLIC`: [cycle start](../../explanation/timing-model.md#rule-2); `R_CYCLIC_META`: [cycle start](../../explanation/timing-model.md#rule-2); `R_META`: [cycle start](../../explanation/timing-model.md#rule-2); `R_MASK`: [cycle start](../../explanation/timing-model.md#rule-2); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `qf` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `MULT.RC.VE` — RC Vector × Scalar Multiply

**Syntax:** `MULT.RC.VE rc_idx src mask_offset mask_shift ds qf`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](operand-types.md#lridx) | `rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`). |
| `src` | [`LcrIdx`](operand-types.md#lcridx) | `src`: **`LR0`**…**`LR15`** \| **`CR0`**…**`CR15`** — if an LR, its stored value selects the scalar from R0/R1 (0..127 → `R0[idx]`, 128..255 → `R1[idx - 128]`, decoded with `qf` and that row's metadata); if a CR, the CR is a MULT constant (NX8: the constant layout, with its own scale and type; FP32: float32 bits; CR0 = 0.0, CR1 = 1.0; see `CrIdx` and [CR scalars](../../explanation/element-formats.md#cr-scalars)). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | `ds`: **`CR0.L`**…**`CR15.H`** — DS half-CR (`ipu_isa.cr_halves`): `num_partitions` builds the partition vectors for `mask_shift`, `mask_type` is the value written into deactivated elements (0, +inf or −inf; NaN is an emulator error). The whole DS is validated. |
| `qf` | [`HalfCrIdx`](operand-types.md#halfcridx) | `qf`: **`CR0.L`**…**`CR15.H`** — QF half-CR naming the NX8 format of the vector elements: in NX8 mode each element expands to FP32 with its row's metadata (`ARCH-stage-mult.md` §3.2.1); ignored in FP32 mode. |

**General description:**
Multiply R_CYCLIC[rc_idx:rc_idx+128] by a scalar, element-wise. The scalar is either a single element of R0/R1 (selected by an LR value) or a CR constant. The partition used by mask_shift and the mask pad come from the DS half `ds`; in NX8 mode the vector elements are decoded with the QF half `qf`.

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the DS `mask_type` value (0, +inf or −inf) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`For each element i: MULT_RES[i] = expand(R_CYCLIC[(rc_idx + i) % 512]) * src_value; then apply mask and shift using DS[ds].num_partitions and DS[ds].mask_type.`

**Example of usage:**
```asm
MULT.RC.VE LR0 LR3 0 LR2 CR15.L CR14.L;;
```

**Register reads:** `rc_idx` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `mask_shift` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `src` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `src` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `R`: [cycle start](../../explanation/timing-model.md#rule-2); `R_META`: [cycle start](../../explanation/timing-model.md#rule-2); `R_CYCLIC`: [cycle start](../../explanation/timing-model.md#rule-2); `R_CYCLIC_META`: [cycle start](../../explanation/timing-model.md#rule-2); `R_MASK`: [cycle start](../../explanation/timing-model.md#rule-2); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `qf` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `MULT.RC.VS` — RC Vector Self-Multiply (Square)

**Syntax:** `MULT.RC.VS rc_idx mask_offset mask_shift ds qf`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](operand-types.md#lridx) | `rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | `ds`: **`CR0.L`**…**`CR15.H`** — DS half-CR (`ipu_isa.cr_halves`): `num_partitions` builds the partition vectors for `mask_shift`, `mask_type` is the value written into deactivated elements (0, +inf or −inf; NaN is an emulator error). The whole DS is validated. |
| `qf` | [`HalfCrIdx`](operand-types.md#halfcridx) | `qf`: **`CR0.L`**…**`CR15.H`** — QF half-CR naming the NX8 format of the vector elements: in NX8 mode each element expands to FP32 with its row's metadata (`ARCH-stage-mult.md` §3.2.1); ignored in FP32 mode. |

**General description:**
Square R_CYCLIC[rc_idx:rc_idx+128] element-wise. The partition used by mask_shift and the mask pad come from the DS half `ds`; in NX8 mode the elements are decoded with the QF half `qf`.

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the DS `mask_type` value (0, +inf or −inf) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`For each element i: rb = expand(R_CYCLIC[(rc_idx + i) % 512]); MULT_RES[i] = rb * rb; then apply mask and shift using DS[ds].num_partitions and DS[ds].mask_type.`

**Example of usage:**
```asm
MULT.RC.VS LR0 0 LR2 CR15.L CR14.L;;
```

**Register reads:** `rc_idx` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `mask_shift` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `R_CYCLIC`: [cycle start](../../explanation/timing-model.md#rule-2); `R_CYCLIC_META`: [cycle start](../../explanation/timing-model.md#rule-2); `R_MASK`: [cycle start](../../explanation/timing-model.md#rule-2); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `qf` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `MULT.VE` — Ra Vector × CR Scalar Multiply

**Syntax:** `MULT.VE ra_idx cr_idx mask_offset mask_shift ds qf`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `ra_idx` | [`LrIdx`](operand-types.md#lridx) | `ra_idx`: **`LR0`**…**`LR15`** — base element offset into combined Ra (`R0` ++ `R1`, 256 elements, cyclic mod 256). |
| `cr_idx` | [`CrIdx`](operand-types.md#cridx) | `cr_idx`: **`CR0`**…**`CR15`** — the scalar: a MULT constant (NX8: `value[7:0]`, `scale[15:8]`, `fe[19:16]`, `sign[23:20]`, decoded with its own scale; FP32: float32 bits). CR0 reads as 0.0 and CR1 as 1.0 in both modes (see `CrIdx` and [CR scalars](../../explanation/element-formats.md#cr-scalars)). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | `ds`: **`CR0.L`**…**`CR15.H`** — DS half-CR (`ipu_isa.cr_halves`): `num_partitions` builds the partition vectors for `mask_shift`, `mask_type` is the value written into deactivated elements (0, +inf or −inf; NaN is an emulator error). The whole DS is validated. |
| `qf` | [`HalfCrIdx`](operand-types.md#halfcridx) | `qf`: **`CR0.L`**…**`CR15.H`** — QF half-CR naming the NX8 format of the vector elements: in NX8 mode each element expands to FP32 with its row's metadata (`ARCH-stage-mult.md` §3.2.1); ignored in FP32 mode. |

**General description:**
Multiply Ra[ra_idx:ra_idx+128] (combined R0/R1) by a CR constant, element-wise. The partition used by mask_shift and the mask pad come from the DS half `ds`; in NX8 mode the Ra elements are decoded with the QF half `qf`.

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the DS `mask_type` value (0, +inf or −inf) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`For each element i: ra = expand(Ra[(ra_idx + i) % 256]); MULT_RES[i] = ra * constant(CR[cr_idx]); then apply mask and shift using DS[ds].num_partitions and DS[ds].mask_type.`

**Example of usage:**
```asm
MULT.VE LR0 CR3 0 LR2 CR15.L CR14.L;;
```

**Register reads:** `ra_idx` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `mask_shift` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `cr_idx` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `R`: [cycle start](../../explanation/timing-model.md#rule-2); `R_META`: [cycle start](../../explanation/timing-model.md#rule-2); `R_MASK`: [cycle start](../../explanation/timing-model.md#rule-2); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `qf` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

### `MULT.EE` — Ra Element × CR Scalar Multiply (broadcast)

**Syntax:** `MULT.EE ra_idx cr_idx mask_offset mask_shift ds qf`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `ra_idx` | [`LrIdx`](operand-types.md#lridx) | `ra_idx`: **`LR0`**…**`LR15`** — index of the single Ra element to read (combined `R0` ++ `R1`, 256 elements, mod 256). |
| `cr_idx` | [`CrIdx`](operand-types.md#cridx) | `cr_idx`: **`CR0`**…**`CR15`** — the scalar: a MULT constant (NX8: `value[7:0]`, `scale[15:8]`, `fe[19:16]`, `sign[23:20]`, decoded with its own scale; FP32: float32 bits). CR0 reads as 0.0 and CR1 as 1.0 in both modes (see `CrIdx` and [CR scalars](../../explanation/element-formats.md#cr-scalars)). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | `ds`: **`CR0.L`**…**`CR15.H`** — DS half-CR (`ipu_isa.cr_halves`): `num_partitions` builds the partition vectors for `mask_shift`, `mask_type` is the value written into deactivated elements (0, +inf or −inf; NaN is an emulator error). The whole DS is validated. |
| `qf` | [`HalfCrIdx`](operand-types.md#halfcridx) | `qf`: **`CR0.L`**…**`CR15.H`** — QF half-CR naming the NX8 format of the vector elements: in NX8 mode each element expands to FP32 with its row's metadata (`ARCH-stage-mult.md` §3.2.1); ignored in FP32 mode. |

**General description:**
Multiply a single Ra element by a CR constant; broadcast the product to all 128 elements. The partition used by mask_shift and the mask pad come from the DS half `ds`; in NX8 mode the Ra element is decoded with the QF half `qf`.

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the DS `mask_type` value (0, +inf or −inf) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`ra = expand(Ra[ra_idx % 256]); result = ra * constant(CR[cr_idx]); for each element i: MULT_RES[i] = result; then apply mask and shift using DS[ds].num_partitions and DS[ds].mask_type.`

**Example of usage:**
```asm
MULT.EE LR0 CR3 0 LR2 CR15.L CR14.L;;
```

**Register reads:** `ra_idx` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `mask_shift` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `cr_idx` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `R`: [cycle start](../../explanation/timing-model.md#rule-2); `R_META`: [cycle start](../../explanation/timing-model.md#rule-2); `R_MASK`: [cycle start](../../explanation/timing-model.md#rule-2); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `qf` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order).

---

## ACC Instructions

Accumulation instructions for combining values with optional masking and shifting.

### `ACC.ADD` — Accumulate Add

**Syntax:** `ACC.ADD`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Running add accumulation: add multiply result into each R_ACC element.

**Pseudo code:**
`R_ACC[i] += MULT_RES[i]  # for each element i`

**Example of usage:**
```asm
ACC.ADD;;
```

**Register reads:** `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3); `R_ACC`: [cycle start](../../explanation/timing-model.md#slot-order).

### `ACC.ADD.FIRST` — Accumulate Add (First)

**Syntax:** `ACC.ADD.FIRST`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Clean-init add: overwrite each R_ACC element with the multiply result (ignores previous R_ACC).

**Pseudo code:**
`R_ACC[i] = MULT_RES[i]  # for each element i`

**Example of usage:**
```asm
ACC.ADD.FIRST;;
```

**Register reads:** `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3).

### `ACC.MAX` — Accumulate Max

**Syntax:** `ACC.MAX`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Running max accumulation: each R_ACC element takes the max of its current value and the multiply result.

**Pseudo code:**
`R_ACC[i] = max(R_ACC[i], MULT_RES[i])  # for each element i`

**Example of usage:**
```asm
ACC.MAX;;
```

**Register reads:** `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3); `R_ACC`: [cycle start](../../explanation/timing-model.md#slot-order).

### `ACC.MAX.FIRST` — Accumulate Max (First)

**Syntax:** `ACC.MAX.FIRST`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Clean-init max: overwrite each R_ACC element unconditionally with the multiply result.

**Pseudo code:**
`R_ACC[i] = MULT_RES[i]  # for each element i (unconditional overwrite)`

**Example of usage:**
```asm
ACC.MAX.FIRST;;
```

**Register reads:** `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3).

### `ACC.SUB` — Accumulate Subtract

**Syntax:** `ACC.SUB`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Running subtract accumulation: subtract the multiply result from each R_ACC element each cycle.

**Pseudo code:**
`R_ACC[i] -= MULT_RES[i]  # for each element i`

**Example of usage:**
```asm
ACC.SUB;;
```

**Register reads:** `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3); `R_ACC`: [cycle start](../../explanation/timing-model.md#slot-order).

### `ACC.SUB.FIRST` — Accumulate Subtract (First)

**Syntax:** `ACC.SUB.FIRST`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Clean-init subtract: set each R_ACC element to the negated multiply result.

**Pseudo code:**
`R_ACC[i] = -MULT_RES[i]  # for each element i`

**Example of usage:**
```asm
ACC.SUB.FIRST;;
```

**Register reads:** `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3).

### `ACC.STRIDE` — Accumulator Stride

**Syntax:** `ACC.STRIDE elements_in_row horizontal_stride vertical_stride offset`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-1 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `elements_in_row` | [`ElementsInRow`](operand-types.md#elementsinrow) | elements_in_row: Elements per row (16, 32, or 64; minimum is 16) |
| `horizontal_stride` | [`HorizontalStride`](operand-types.md#horizontalstride) | horizontal_stride: Horizontal stride mode (off, on, on_inv) |
| `vertical_stride` | [`VerticalStride`](operand-types.md#verticalstride) | vertical_stride: Vertical stride mode (enabled, inverted) |
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: LR register; value % 4 gives start index in RACC (0, 32, 64, or 96) |

**General description:**
Reorder the multiplication result into R_ACC using horizontal/vertical stride decimation. Only updates the RACC indexes written; leaves the rest unchanged.

**Pseudo code:**
`Decimate MULT_RES as rows×cols; apply horizontal stride (take every 2nd column); then vertical stride (take every 2nd row). Write result into R_ACC[start:start+N] where start = (offset%4)*32, N = 32|64|128. When a data structure has fewer than 8 elements, hardware pads to 16 automatically (not programmable).`

**Example of usage:**
```asm
ACC.STRIDE 16 off off LR0;;
```

**Register reads:** `offset` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3).

### `AGG.SUM.FIRST` — Aggregate Sum (First)

**Syntax:** `AGG.SUM.FIRST dest_slot ds`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](operand-types.md#lridx) | dest_slot: LR register whose value gives the destination slot in R_ACC (0–127) |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | ds: CR0.L…CR15.H, the DS half supplying valid_elements (the whole DS is validated) |

**General description:**
Sum active MULT_RES elements and write the result into R_ACC at the slot given by LR. The current value at the destination slot is NOT included in the sum (clean initialisation). Only the active element count (``valid_elements``) of the DS half ``ds`` is used: the first ``valid_elements`` elements (ARCH-stage-acc).

**Pseudo code:**
`Let n = DS[ds].valid_elements. dest = LR[dest_slot] % 128. R_ACC[dest] = sum(MULT_RES[0..n-1]).`

**Example of usage:**
```asm
AGG.SUM.FIRST LR0 CR3.L;;
```

**Register reads:** `dest_slot` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3).

### `AGG.SUM` — Aggregate Sum

**Syntax:** `AGG.SUM dest_slot ds`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](operand-types.md#lridx) | dest_slot: LR register whose value gives the destination slot in R_ACC (0–127) |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | ds: CR0.L…CR15.H, the DS half supplying valid_elements (the whole DS is validated) |

**General description:**
Sum active MULT_RES elements and ADD the result to R_ACC at the slot given by LR (running cross-cycle accumulation). Only the active element count (``valid_elements``) of the DS half ``ds`` is used: the first ``valid_elements`` elements (ARCH-stage-acc).

**Pseudo code:**
`Let n = DS[ds].valid_elements. dest = LR[dest_slot] % 128. R_ACC[dest] = sum(MULT_RES[0..n-1]) + R_ACC[dest].`

**Example of usage:**
```asm
AGG.SUM LR0 CR3.L;;
```

**Register reads:** `dest_slot` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3); `R_ACC`: [cycle start](../../explanation/timing-model.md#slot-order).

### `AGG.MAX.FIRST` — Aggregate Max (First)

**Syntax:** `AGG.MAX.FIRST dest_slot ds`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](operand-types.md#lridx) | dest_slot: LR register whose value gives the destination slot in R_ACC (0–127) |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | ds: CR0.L…CR15.H, the DS half supplying valid_elements (the whole DS is validated) |

**General description:**
Find the maximum of active MULT_RES elements and write it into R_ACC at the slot given by LR. The current value at the destination slot is NOT used as a seed (clean initialisation). Only the active element count (``valid_elements``) of the DS half ``ds`` is used: the first ``valid_elements`` elements (ARCH-stage-acc).

**Pseudo code:**
`Let n = DS[ds].valid_elements. dest = LR[dest_slot] % 128. R_ACC[dest] = max(MULT_RES[0..n-1]); when n = 0 the identity seed (-inf) is written.`

**Example of usage:**
```asm
AGG.MAX.FIRST LR0 CR3.L;;
```

**Register reads:** `dest_slot` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3).

### `AGG.MAX` — Aggregate Max

**Syntax:** `AGG.MAX dest_slot ds`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](operand-types.md#lridx) | dest_slot: LR register whose value gives the destination slot in R_ACC (0–127) |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | ds: CR0.L…CR15.H, the DS half supplying valid_elements (the whole DS is validated) |

**General description:**
Find the maximum of active MULT_RES elements seeded with the current destination slot value (running cross-cycle max). Only the active element count (``valid_elements``) of the DS half ``ds`` is used: the first ``valid_elements`` elements (ARCH-stage-acc).

**Pseudo code:**
`Let n = DS[ds].valid_elements. dest = LR[dest_slot] % 128. R_ACC[dest] = max(MULT_RES[0..n-1], R_ACC[dest]).`

**Example of usage:**
```asm
AGG.MAX LR0 CR3.L;;
```

**Register reads:** `dest_slot` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3); `R_ACC`: [cycle start](../../explanation/timing-model.md#slot-order).

### `ACC.RESHAPE` — Reshape

**Syntax:** `ACC.RESHAPE source dest reshape_mask`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-5 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `source` | [`LrdIdx`](operand-types.md#lrdidx) | source: LRDn register pair (LRD0, LRD2, ..., LRD14) read as source[0..7] — 8 byte elements, each a MULT_RES word index |
| `dest` | [`LrdIdx`](operand-types.md#lrdidx) | dest: LRDn register pair (LRD0, LRD2, ..., LRD14) read as dest[0..7] — 8 byte elements, each an R_ACC word index |
| `reshape_mask` | [`LrOrReshapeMaskImmediate`](operand-types.md#lrorreshapemaskimmediate) | reshape_mask: immediate 0-7 or an LR register; selects the first participating element index (elements reshape_mask..7 participate; reshape_mask = 0 uses all 8, reshape_mask = 7 uses only element 7). When an LR is given, its full value is used as the mask at runtime. |

**General description:**
Scatter MULT_RES elements into R_ACC: for each of the trailing (8 - reshape_mask) byte elements of the source/dest LRDn pairs (indices reshape_mask..7), copy MULT_RES[source[i]] to R_ACC[dest[i]].

> reshape_mask skips the lowest-indexed elements 0..reshape_mask-1; the rest participate. Example: reshape_mask = 2 skips elements 0, 1; elements 2-7 participate. Example: reshape_mask = 5 skips elements 0-4; elements 5-7 participate.

**Pseudo code:**
`mult_res = MULT_RES (this cycle's MULT result)
mask = reshape_mask if immediate else LR[reshape_mask]
if mask > 8: raise error
for i in mask..7:
    if source[i] >= 128 or dest[i] >= 128: raise error
    R_ACC[dest[i]] = mult_res[source[i]]`

**Example of usage:**
```asm
ACC.RESHAPE LRD0 LRD2 0;;
```

**Register reads:** `source` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `dest` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `reshape_mask` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `MULT_RES`: [same-cycle forward](../../explanation/timing-model.md#rule-3).

---

## AAQ Instructions

Activation and quantization: `ACTIVATE.QUANTIZE` activates the forwarded R_ACC, quantizes it to one NX8 row and stores the row and its metadata to XMEM; `LD.LUT` loads one LUT segment from XMEM (ADR 0007).

### `LD.LUT` — Load LUT Segment

**Syntax:** `LD.LUT lut_addr table_id table_offset`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-1, F-6, F-7 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `lut_addr` | [`LutAddrImmediate`](operand-types.md#lutaddrimmediate) | lut_addr: immediate segment index 0–4 (3-bit field; 5–7 are reserved) |
| `table_id` | [`CrIdx`](operand-types.md#cridx) | table_id: CR0–CR15 holding the base row |
| `table_offset` | [`LrIdx`](operand-types.md#lridx) | table_offset: LR0–LR15 holding the row offset (the row is CR[table_id] + LR[table_offset]) |

**General description:**
Load the first 128 bytes (1024 bits) of one XMEM row into LUT segment `lut_addr` of the AaQ stage. Five loads (segments 0-4) fill the 4352 table bits and the 46-bit LUT metadata the `generic` activation reads. No row metadata, no scaling; the same in both modes.

> The architecture specs read the LUT from `R_ACC`; the emulator reads XMEM (ADR 0007, OQ-39). A word with `LD.LUT` and any `LD.*` of the LOAD slot is an assembler error: they share the XMEM read port.

**Pseudo code:**
`row = CR[table_id] + LR[table_offset]
LUT.segment[lut_addr] = Memory[row][0 .. 127]  # 1024 bits, both modes`

**Example of usage:**
```asm
LD.LUT 0 CR4 LR0;;
```

**Register reads:** `table_id` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `table_offset` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1).

### `ACTIVATE.QUANTIZE` — Activate, Quantize and Store

**Syntax:** `ACTIVATE.QUANTIZE activation_fn ds qf table_id table_offset`

**Tag:** `HARDWARE` (implemented by the IPU)

**Faults:** F-1, F-6, F-7 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `activation_fn` | [`AaqActivationFn`](operand-types.md#aaqactivationfn) | activation_fn: identity, relu, relu6, reciprocal, rsqrt, exp2 or generic (3-bit ARCH encoding) |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | ds: CR0.L–CR15.H, the DS half naming the active elements |
| `qf` | [`HalfCrIdx`](operand-types.md#halfcridx) | qf: CR0.L–CR15.H, the QF half naming the NX8 output format |
| `table_id` | [`CrIdx`](operand-types.md#cridx) | table_id: CR0–CR15 holding the base row |
| `table_offset` | [`LrIdx`](operand-types.md#lridx) | table_offset: LR0–LR15 holding the row offset |

**General description:**
Activate the forwarded R_ACC (this cycle's ACC result), quantize the active elements to one NX8 row in the format the QF half `qf` names, and store the 128-byte row and its metadata (scale, dynamic exponent) at XMEM row CR[table_id] + LR[table_offset]. Inactive elements are written 0. In FP32 mode the activated row is stored as 512 bytes of FP32, inactive lanes 0, and `qf` is ignored. R_ACC is not modified.

> Special cases follow the architecture spec: reciprocal and rsqrt of a subnormal input add the smallest normal; rsqrt of a negative input is NaN, which cannot be quantized (emulator error); exp2 of a positive input is an emulator error. ±inf saturates to the largest magnitude; a saturation sets QUANT_SATURATION. `generic` computes the function the test declares as loaded with `LD.LUT`.

**Pseudo code:**
`y[i] = activation(R_ACC[i]) for the active elements i of DS[ds]
NX8: (bytes, scale, dynamic_exponent) = quantize(y, QF[qf])  # ARCH-stage-aaq-str §4.3.2
     Memory[row] = bytes; Metadata[row] = (scale, dynamic_exponent)
FP32: Memory[row] = y as 128 FP32 (inactive lanes 0)
where row = CR[table_id] + LR[table_offset]`

**Example of usage:**
```asm
ACTIVATE.QUANTIZE relu CR15.L CR14.L CR2 LR0;;
```

**Register reads:** `table_id` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `table_offset` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `qf` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `R_ACC`: [same-cycle forward](../../explanation/timing-model.md#rule-3); `AAQ_LUT`: [cycle start](../../explanation/timing-model.md#slot-order).

---

## AAQ_EXACT Instructions

> **Emulator-only slot** — not implemented in real IPU hardware: not in the hardware word or the SV package.

Emulator-only slot: `ACTIVATE.EXACT` applies the exact activation to the forwarded R_ACC and stores one FP32 row (512 B). Valid in FP32 mode only (ADR 0007).

### `ACTIVATE.EXACT` — Activate Exactly and Store FP32

**Syntax:** `ACTIVATE.EXACT activation_fn ds table_id table_offset`

**Tag:** `EMULATOR` (emulator only; not in the hardware word or the SV package)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `activation_fn` | [`ActivationFn`](operand-types.md#activationfn) | activation_fn: one of identity, relu, relu6, sigmoid, tanh, gelu, softplus, elu, exp2, reciprocal, rsqrt, silu, window (see ACTIVATION_FN_NAMES) |
| `ds` | [`HalfCrIdx`](operand-types.md#halfcridx) | ds: CR0.L–CR15.H, the DS half naming the active elements |
| `table_id` | [`CrIdx`](operand-types.md#cridx) | table_id: CR0–CR15 holding the base row |
| `table_offset` | [`LrIdx`](operand-types.md#lridx) | table_offset: LR0–LR15 holding the row offset |

**General description:**
Emulator-only. Apply the exact Python activation to the active elements of the forwarded R_ACC and store the row as 128 FP32 values (512 bytes) at XMEM row CR[table_id] + LR[table_offset]; inactive lanes are written 0. Valid in FP32 mode only: a program that uses it is rejected when loaded into an NX8-mode IPU. R_ACC is not modified.

**Pseudo code:**
`row = CR[table_id] + LR[table_offset]
Memory[row] = activation(R_ACC[i]) as FP32 for active i, 0 for inactive i
rsqrt(x) = 0 for x <= 0; window(x) = 1 for a <= x < b, else 0. α for elu and the window bounds are emulator settings (docs/content/reference/isa/activate-quantize.md).`

**Example of usage:**
```asm
ACTIVATE.EXACT relu CR15.L CR2 LR0;;
```

**Register reads:** `table_id` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `table_offset` (`LR`): [after LR writes](../../explanation/timing-model.md#rule-1); `ds` (`CR`): [cycle start](../../explanation/timing-model.md#slot-order); `R_ACC`: [same-cycle forward](../../explanation/timing-model.md#slot-order).

---

## Break Instructions

> **Emulator-only slot** — not implemented in real IPU hardware: not in the hardware word or the SV package.

Debug break instructions for halting execution and entering debug mode.

### `BREAK` — Break

**Syntax:** `BREAK`

**Tag:** `EMULATOR` (emulator only; not in the hardware word or the SV package)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Unconditional break.

**Pseudo code:**
`Halt execution`


### `BREAK.IFEQ` — Break if Equal

**Syntax:** `BREAK.IFEQ reg value`

**Tag:** `EMULATOR` (emulator only; not in the hardware word or the SV package)

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LrIdx`](operand-types.md#lridx) | reg: Register to test (LR0–LR15) |
| `value` | [`BreakImmediate`](operand-types.md#breakimmediate) | value: Immediate value to compare against |

**General description:**
Break execution if register equals value.

**Pseudo code:**
`if (reg == value) BREAK`

**Example of usage:**
```asm
BREAK.IFEQ LR0 10;;
```

**Register reads:** `reg` (`LR`): [cycle start](../../explanation/timing-model.md#rule-1).

---
