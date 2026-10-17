# IPU Assembly Instruction Reference

Per-opcode documentation is generated from `InstructionDoc` entries in `instruction_spec.py`. Operand **types** link to the shared [operand type reference](operand-types.md).

## Slots

A VLIW instruction word encodes one sub-instruction per slot. Slots are grouped into pipeline stages that execute sequentially within a cycle: **CTRL** (COND + LR run concurrently; LOAD address is resolved here and the data feeds MULT) → **MULT** → **ACC** → **AAQ** → **STORE**. `ACC_STORE`, `BREAK` are emulator-only: not in the hardware word. Any omitted slot is filled with `NOP` by the assembler automatically.

| Slot | Count | Tag | Description |
|------|------:|-----|-------------|
| `COND` | 1 | `HARDWARE` | Conditional and unconditional branches; control flow based on register comparisons. |
| `LR` | 3 | `HARDWARE` | Loop register instructions for controlling addresses, counters, and scalars. Three independent sub-slots per cycle. |
| `LOAD` | 1 | `HARDWARE` | First-stage memory loads that feed the multiply unit (mult-stage, cyclic, and mask registers). |
| `MULT` | 1 | `HARDWARE` | Multiply-stage instructions: element-wise and vector multiply operations. |
| `ACC` | 1 | `HARDWARE` | Accumulation instructions for combining multiply results into R_ACC. |
| `AAQ` | 1 | `HARDWARE` | Activation and quantization: apply activation to R_ACC and write quantized INT8 bytes to POST_AAQ_REG. |
| `STORE` | 1 | `HARDWARE` | Last-stage memory stores that drain POST_AAQ_REG to external memory after activation and quantization. |
| `ACC_STORE` | 1 | `EMULATOR` | Simulation-only stores of R_ACC to external memory (ST.ACC). Not implemented in real IPU hardware. |
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

### Hardware word (155 bits)

The part of the instruction word the IPU decodes: the hardware slots
(`COND`, `LR`×3, `LOAD`, `MULT`, `ACC`, `AAQ`, `STORE`). The SystemVerilog package describes this
word (`IPU_COMPOUND_INST_WIDTH = 155`).

#### Whole-word bit layout

<svg width="888" height="641" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .inst-title { font-size: 14px; font-weight: bold; font-family: Arial; }
      .inst-label { font-size: 8px; font-weight: bold; font-family: Arial; }
      .field-label { font-size: 7px; font-family: Arial; }
      .bit-range { font-size: 8px; font-family: Arial; }
    </style>
  </defs>
  <rect x="0" y="0" width="888" height="641" fill="white" stroke="none"/>
  <text x="444.0" y="34" text-anchor="middle" class="inst-title">Hardware word Layout - 155 total bits</text>
  <rect x="80" y="55.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="99.0" text-anchor="end" class="bit-range">[154:128]</text>
  <text x="212.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">154</text>
  <text x="236.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">153</text>
  <text x="260.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">152</text>
  <text x="284.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">151</text>
  <text x="308.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">150</text>
  <text x="332.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">149</text>
  <text x="356.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">148</text>
  <text x="380.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">147</text>
  <text x="404.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">146</text>
  <text x="428.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">145</text>
  <text x="452.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">144</text>
  <text x="476.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">143</text>
  <text x="500.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">142</text>
  <text x="524.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">141</text>
  <text x="548.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">140</text>
  <text x="572.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">139</text>
  <text x="596.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">138</text>
  <text x="620.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">137</text>
  <text x="644.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">136</text>
  <text x="668.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">135</text>
  <text x="692.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">134</text>
  <text x="716.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">133</text>
  <text x="740.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">132</text>
  <text x="764.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">131</text>
  <text x="788.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">130</text>
  <text x="812.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">129</text>
  <text x="836.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">128</text>
  <rect x="752" y="55.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="78.5" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="89.5" text-anchor="middle" class="field-label">inst</text>
  <text x="800.0" y="100.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="800.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[131:128]</text>
  <rect x="632" y="55.0" width="120" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="692.0" y="84.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="692.0" y="95.0" text-anchor="middle" class="field-label">reg</text>
  <text x="692.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[136:132]</text>
  <rect x="512" y="55.0" width="120" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="572.0" y="84.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="572.0" y="95.0" text-anchor="middle" class="field-label">reg</text>
  <text x="572.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[141:137]</text>
  <rect x="272" y="55.0" width="240" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="392.0" y="84.0" text-anchor="middle" class="field-label">label</text>
  <text x="392.0" y="95.0" text-anchor="middle" class="field-label">token</text>
  <text x="392.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[151:142]</text>
  <rect x="200" y="55.0" width="72" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="236.0" y="78.5" text-anchor="middle" class="field-label">cond</text>
  <text x="236.0" y="89.5" text-anchor="middle" class="field-label">inst</text>
  <text x="236.0" y="100.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="236.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[154:152]</text>
  <rect x="80" y="135.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="179.0" text-anchor="end" class="bit-range">[127:96]</text>
  <text x="92.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">127</text>
  <text x="116.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">126</text>
  <text x="140.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">125</text>
  <text x="164.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">124</text>
  <text x="188.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">123</text>
  <text x="212.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">122</text>
  <text x="236.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">121</text>
  <text x="260.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">120</text>
  <text x="284.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">119</text>
  <text x="308.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">118</text>
  <text x="332.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">117</text>
  <text x="356.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">116</text>
  <text x="380.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">115</text>
  <text x="404.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">114</text>
  <text x="428.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">113</text>
  <text x="452.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">112</text>
  <text x="476.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">111</text>
  <text x="500.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">110</text>
  <text x="524.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">109</text>
  <text x="548.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">108</text>
  <text x="572.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">107</text>
  <text x="596.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">106</text>
  <text x="620.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">105</text>
  <text x="644.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">104</text>
  <text x="668.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">103</text>
  <text x="692.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">102</text>
  <text x="716.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">101</text>
  <text x="740.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">100</text>
  <text x="764.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">99</text>
  <text x="788.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">98</text>
  <text x="812.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">97</text>
  <text x="836.0" y="147.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">96</text>
  <rect x="752" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="800.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[99:96]</text>
  <rect x="560" y="135.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="656.0" y="164.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="656.0" y="175.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="656.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[107:100]</text>
  <rect x="464" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="512.0" y="158.5" text-anchor="middle" class="field-label">lr</text>
  <text x="512.0" y="169.5" text-anchor="middle" class="field-label">inst</text>
  <text x="512.0" y="180.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="512.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[111:108]</text>
  <rect x="368" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="416.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="416.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="416.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[115:112]</text>
  <rect x="272" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="320.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="320.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="320.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[119:116]</text>
  <rect x="80" y="135.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="176.0" y="164.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="176.0" y="175.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="176.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[127:120]</text>
  <rect x="80" y="215.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="259.0" text-anchor="end" class="bit-range">[95:64]</text>
  <text x="92.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">95</text>
  <text x="116.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">94</text>
  <text x="140.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">93</text>
  <text x="164.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">92</text>
  <text x="188.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">91</text>
  <text x="212.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">90</text>
  <text x="236.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">89</text>
  <text x="260.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">88</text>
  <text x="284.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">87</text>
  <text x="308.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">86</text>
  <text x="332.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">85</text>
  <text x="356.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">84</text>
  <text x="380.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">83</text>
  <text x="404.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">82</text>
  <text x="428.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">81</text>
  <text x="452.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">80</text>
  <text x="476.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">79</text>
  <text x="500.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">78</text>
  <text x="524.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">77</text>
  <text x="548.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">76</text>
  <text x="572.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">75</text>
  <text x="596.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">74</text>
  <text x="620.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">73</text>
  <text x="644.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">72</text>
  <text x="668.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">71</text>
  <text x="692.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">70</text>
  <text x="716.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">69</text>
  <text x="740.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">68</text>
  <text x="764.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">67</text>
  <text x="788.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">66</text>
  <text x="812.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">65</text>
  <text x="836.0" y="227.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">64</text>
  <rect x="800" y="215.0" width="48" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="824.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="824.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="824.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[65:62]</text>
  <rect x="704" y="215.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="752.0" y="244.0" text-anchor="middle" class="field-label">cr</text>
  <text x="752.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="752.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[69:66]</text>
  <rect x="656" y="215.0" width="48" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="680.0" y="238.5" text-anchor="middle" class="field-label">load</text>
  <text x="680.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="680.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="680.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[71:70]</text>
  <rect x="560" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="608.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="608.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="608.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[75:72]</text>
  <rect x="464" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="512.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="512.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="512.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[79:76]</text>
  <rect x="272" y="215.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="368.0" y="244.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="368.0" y="255.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="368.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[87:80]</text>
  <rect x="176" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="224.0" y="238.5" text-anchor="middle" class="field-label">lr</text>
  <text x="224.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="224.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="224.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[91:88]</text>
  <rect x="80" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="128.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="128.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="128.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[95:92]</text>
  <rect x="80" y="295.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="339.0" text-anchor="end" class="bit-range">[63:32]</text>
  <text x="92.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">63</text>
  <text x="116.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">62</text>
  <text x="140.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">61</text>
  <text x="164.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">60</text>
  <text x="188.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">59</text>
  <text x="212.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">58</text>
  <text x="236.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">57</text>
  <text x="260.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">56</text>
  <text x="284.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">55</text>
  <text x="308.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">54</text>
  <text x="332.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">53</text>
  <text x="356.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">52</text>
  <text x="380.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">51</text>
  <text x="404.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">50</text>
  <text x="428.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">49</text>
  <text x="452.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">48</text>
  <text x="476.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">47</text>
  <text x="500.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">46</text>
  <text x="524.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">45</text>
  <text x="548.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">44</text>
  <text x="572.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">43</text>
  <text x="596.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">42</text>
  <text x="620.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">41</text>
  <text x="644.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">40</text>
  <text x="668.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">39</text>
  <text x="692.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">38</text>
  <text x="716.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">37</text>
  <text x="740.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">36</text>
  <text x="764.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">35</text>
  <text x="788.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">34</text>
  <text x="812.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">33</text>
  <text x="836.0" y="307.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">32</text>
  <rect x="776" y="295.0" width="72" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="812.0" y="318.5" text-anchor="middle" class="field-label">acc</text>
  <text x="812.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="812.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="812.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[34:31]</text>
  <rect x="704" y="295.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="740.0" y="313.0" text-anchor="middle" class="field-label">mult</text>
  <text x="740.0" y="324.0" text-anchor="middle" class="field-label">mask</text>
  <text x="740.0" y="335.0" text-anchor="middle" class="field-label">offset</text>
  <text x="740.0" y="346.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="740.0" y="360.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[37:35]</text>
  <rect x="608" y="295.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="656.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="656.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="656.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[41:38]</text>
  <rect x="512" y="295.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="560.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="560.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="560.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[45:42]</text>
  <rect x="416" y="295.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="464.0" y="318.5" text-anchor="middle" class="field-label">dstructure</text>
  <text x="464.0" y="329.5" text-anchor="middle" class="field-label">cr</text>
  <text x="464.0" y="340.5" text-anchor="middle" class="field-label">reg</text>
  <text x="464.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[49:46]</text>
  <rect x="296" y="295.0" width="120" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="356.0" y="324.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="356.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="356.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[54:50]</text>
  <rect x="224" y="295.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="260.0" y="318.5" text-anchor="middle" class="field-label">mult</text>
  <text x="260.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="260.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="260.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[57:55]</text>
  <rect x="128" y="295.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="176.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="176.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="176.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[61:58]</text>
  <rect x="80" y="295.0" width="48" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="104.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="104.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="104.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[65:62]</text>
  <rect x="80" y="375.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="419.0" text-anchor="end" class="bit-range">[31:0]</text>
  <text x="92.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">31</text>
  <text x="116.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">30</text>
  <text x="140.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">29</text>
  <text x="164.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">28</text>
  <text x="188.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">27</text>
  <text x="212.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">26</text>
  <text x="236.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">25</text>
  <text x="260.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">24</text>
  <text x="284.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">23</text>
  <text x="308.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">22</text>
  <text x="332.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">21</text>
  <text x="356.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">20</text>
  <text x="380.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">19</text>
  <text x="404.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">18</text>
  <text x="428.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">17</text>
  <text x="452.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">16</text>
  <text x="476.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">15</text>
  <text x="500.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">14</text>
  <text x="524.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">13</text>
  <text x="548.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">12</text>
  <text x="572.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">11</text>
  <text x="596.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">10</text>
  <text x="620.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">9</text>
  <text x="644.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">8</text>
  <text x="668.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">7</text>
  <text x="692.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">6</text>
  <text x="716.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">5</text>
  <text x="740.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">4</text>
  <text x="764.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">3</text>
  <text x="788.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">2</text>
  <text x="812.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">1</text>
  <text x="836.0" y="387.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">0</text>
  <rect x="752" y="375.0" width="96" height="80" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="800.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[3:0]</text>
  <rect x="656" y="375.0" width="96" height="80" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="704.0" y="404.0" text-anchor="middle" class="field-label">cr</text>
  <text x="704.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="704.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[7:4]</text>
  <rect x="632" y="375.0" width="24" height="80" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="644.0" y="398.5" text-anchor="middle" class="field-label">store</text>
  <text x="644.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="644.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="644.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[8:8]</text>
  <rect x="536" y="375.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="584.0" y="398.5" text-anchor="middle" class="field-label">dstructure</text>
  <text x="584.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="584.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="584.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[12:9]</text>
  <rect x="440" y="375.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="488.0" y="404.0" text-anchor="middle" class="field-label">activation</text>
  <text x="488.0" y="415.0" text-anchor="middle" class="field-label">fn</text>
  <text x="488.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[16:13]</text>
  <rect x="416" y="375.0" width="24" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="428.0" y="398.5" text-anchor="middle" class="field-label">aaq</text>
  <text x="428.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="428.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="428.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[17:17]</text>
  <rect x="368" y="375.0" width="48" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="392.0" y="404.0" text-anchor="middle" class="field-label">vertical</text>
  <text x="392.0" y="415.0" text-anchor="middle" class="field-label">stride</text>
  <text x="392.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[19:18]</text>
  <rect x="272" y="375.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="320.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="320.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="320.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[23:20]</text>
  <rect x="200" y="375.0" width="72" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="236.0" y="404.0" text-anchor="middle" class="field-label">lrd</text>
  <text x="236.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="236.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[26:24]</text>
  <rect x="104" y="375.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="152.0" y="398.5" text-anchor="middle" class="field-label">dstructure</text>
  <text x="152.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="152.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="152.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[30:27]</text>
  <rect x="80" y="375.0" width="24" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="398.5" text-anchor="middle" class="field-label">acc</text>
  <text x="92.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="92.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="92.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[34:31]</text>
  <text x="80" y="475.0" class="inst-label" style="font-weight: bold;">Instruction Type Colors:</text>
  <rect x="80" y="480.0" width="12" height="12" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="492.0" class="field-label" style="font-size: 9px;">LoadInst (Memory Load)</text>
  <rect x="80" y="495.0" width="12" height="12" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="507.0" class="field-label" style="font-size: 9px;">StoreInst (Memory Store)</text>
  <rect x="80" y="510.0" width="12" height="12" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="522.0" class="field-label" style="font-size: 9px;">MultInst (Multiply)</text>
  <rect x="80" y="525.0" width="12" height="12" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="537.0" class="field-label" style="font-size: 9px;">AccInst (Accumulator)</text>
  <rect x="80" y="540.0" width="12" height="12" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="552.0" class="field-label" style="font-size: 9px;">AaqInst (Activation and Quantization)</text>
  <rect x="80" y="555.0" width="12" height="12" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="567.0" class="field-label" style="font-size: 9px;">LrInst (Link Register)</text>
  <rect x="80" y="570.0" width="12" height="12" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="582.0" class="field-label" style="font-size: 9px;">CondInst (Conditional)</text>
</svg>

#### Per-slot union layout

<svg xmlns="http://www.w3.org/2000/svg" width="660" height="1602" viewBox="0 0 660 1602">
  <rect x="0" y="0" width="660" height="1602" fill="white"/>
  <text x="330" y="38" text-anchor="middle" font-size="14" font-weight="bold" font-family="Arial">IPU VLIW — Union Field Layout (per slot, MSB → LSB)</text>
  <text x="330" y="68" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">COND  (23 bits = 3b opcode + 10+5+5b fields)</text>
  <rect x="24" y="76" width="79" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="63" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="63" y="118" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <rect x="103" y="76" width="266" height="64" fill="#98D8C8" stroke="#333" stroke-width="1" rx="3"/>
  <text x="236" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">Label</text>
  <text x="236" y="118" text-anchor="middle" font-size="8" font-family="Arial">10b</text>
  <rect x="369" y="76" width="133" height="64" fill="#98D8C8" stroke="#333" stroke-width="1" rx="3"/>
  <text x="435" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LcrIdx</text>
  <text x="435" y="118" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="502" y="76" width="134" height="64" fill="#98D8C8" stroke="#333" stroke-width="1" rx="3"/>
  <text x="569" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LcrIdx</text>
  <text x="569" y="118" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <text x="28" y="156" font-size="8.5" font-family="monospace" fill="#222">BEQ</text>
  <rect x="134" y="147" width="218" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="243" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">label</text>
  <rect x="352" y="147" width="109" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="406" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">reg1</text>
  <rect x="461" y="147" width="175" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="548" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">reg2</text>
  <text x="28" y="170" font-size="8.5" font-family="monospace" fill="#222">BGE.S</text>
  <rect x="134" y="161" width="218" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="243" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">label</text>
  <rect x="352" y="161" width="109" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="406" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">reg1</text>
  <rect x="461" y="161" width="175" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="548" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">reg2</text>
  <text x="28" y="184" font-size="8.5" font-family="monospace" fill="#222">BLT.S</text>
  <rect x="134" y="175" width="218" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="243" y="184" text-anchor="middle" font-size="7.5" font-family="Arial">label</text>
  <rect x="352" y="175" width="109" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="406" y="184" text-anchor="middle" font-size="7.5" font-family="Arial">reg1</text>
  <rect x="461" y="175" width="175" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="548" y="184" text-anchor="middle" font-size="7.5" font-family="Arial">reg2</text>
  <text x="28" y="198" font-size="8.5" font-family="monospace" fill="#222">BNE</text>
  <rect x="134" y="189" width="218" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="243" y="198" text-anchor="middle" font-size="7.5" font-family="Arial">label</text>
  <rect x="352" y="189" width="109" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="406" y="198" text-anchor="middle" font-size="7.5" font-family="Arial">reg1</text>
  <rect x="461" y="189" width="175" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="548" y="198" text-anchor="middle" font-size="7.5" font-family="Arial">reg2</text>
  <text x="28" y="212" font-size="8.5" font-family="monospace" fill="#222">BR</text>
  <rect x="134" y="203" width="218" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="243" y="212" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="352" y="203" width="109" height="12" fill="#98D8C855" stroke="#bbb" stroke-width="0.5"/>
  <text x="406" y="212" text-anchor="middle" font-size="7.5" font-family="Arial">reg</text>
  <rect x="461" y="203" width="175" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="548" y="212" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="226" font-size="8.5" font-family="monospace" fill="#222">HALT</text>
  <rect x="134" y="217" width="218" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="243" y="226" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="352" y="217" width="109" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="406" y="226" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="461" y="217" width="175" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="548" y="226" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="240" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="231" width="218" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="243" y="240" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="352" y="231" width="109" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="406" y="240" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="461" y="231" width="175" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="548" y="240" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
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
  <text x="234" y="386" text-anchor="middle" font-size="7.5" font-family="Arial">src_b:LcrIdx</text>
  <rect x="334" y="377" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="386" text-anchor="middle" font-size="7.5" font-family="Arial">dest</text>
  <rect x="434" y="377" width="202" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="386" text-anchor="middle" font-size="7.5" font-family="Arial">src_a</text>
  <text x="28" y="400" font-size="8.5" font-family="monospace" fill="#222">ADDB</text>
  <rect x="134" y="391" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="400" text-anchor="middle" font-size="7.5" font-family="Arial">src_b:LcrIdx</text>
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
  <text x="234" y="456" text-anchor="middle" font-size="7.5" font-family="Arial">step:LcrIdx</text>
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
  <text x="234" y="484" text-anchor="middle" font-size="7.5" font-family="Arial">src:CrIdx</text>
  <rect x="334" y="475" width="100" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="384" y="484" text-anchor="middle" font-size="7.5" font-family="Arial">reg</text>
  <rect x="434" y="475" width="202" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="535" y="484" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="498" font-size="8.5" font-family="monospace" fill="#222">SUB</text>
  <rect x="134" y="489" width="200" height="12" fill="#FFA07A55" stroke="#bbb" stroke-width="0.5"/>
  <text x="234" y="498" text-anchor="middle" font-size="7.5" font-family="Arial">src_b:LcrIdx</text>
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
  <text x="330" y="744" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">MULT  (23 bits = 3b opcode + 5+4+4+4+3b fields)</text>
  <rect x="24" y="752" width="79" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="63" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="63" y="794" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <rect x="103" y="752" width="133" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="169" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LcrIdx</text>
  <text x="169" y="794" text-anchor="middle" font-size="8" font-family="Arial">5b</text>
  <rect x="236" y="752" width="106" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="289" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">DstructureCrIdx</text>
  <text x="289" y="794" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="342" y="752" width="106" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="395" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="395" y="794" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="448" y="752" width="106" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="501" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="501" y="794" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="554" y="752" width="82" height="64" fill="#4ECDC4" stroke="#333" stroke-width="1" rx="3"/>
  <text x="595" y="780" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">MultMaskOffsetImmediate</text>
  <text x="595" y="794" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <text x="28" y="832" font-size="8.5" font-family="monospace" fill="#222">MULT.EE</text>
  <rect x="134" y="823" width="109" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="188" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx:CrIdx</text>
  <rect x="243" y="823" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="286" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">dstructure_cr_idx</text>
  <rect x="329" y="823" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="372" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">ra_idx</text>
  <rect x="415" y="823" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="458" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="501" y="823" width="135" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="568" y="832" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="846" font-size="8.5" font-family="monospace" fill="#222">MULT.RC.VE</text>
  <rect x="134" y="837" width="109" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="188" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">src</text>
  <rect x="243" y="837" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="286" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx</text>
  <rect x="329" y="837" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="372" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">rc_idx</text>
  <rect x="415" y="837" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="458" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="501" y="837" width="135" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="568" y="846" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="860" font-size="8.5" font-family="monospace" fill="#222">MULT.RC.VS</text>
  <rect x="134" y="851" width="109" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="188" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="243" y="851" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="286" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx</text>
  <rect x="329" y="851" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="372" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">rc_idx</text>
  <rect x="415" y="851" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="458" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="501" y="851" width="135" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="568" y="860" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="874" font-size="8.5" font-family="monospace" fill="#222">MULT.RC.VV</text>
  <rect x="134" y="865" width="109" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="188" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">ra:MultStageReg</text>
  <rect x="243" y="865" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="286" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx</text>
  <rect x="329" y="865" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="372" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">rc_idx</text>
  <rect x="415" y="865" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="458" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="501" y="865" width="135" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="568" y="874" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="888" font-size="8.5" font-family="monospace" fill="#222">MULT.VE</text>
  <rect x="134" y="879" width="109" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="188" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx:CrIdx</text>
  <rect x="243" y="879" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="286" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">dstructure_cr_idx</text>
  <rect x="329" y="879" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="372" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">ra_idx</text>
  <rect x="415" y="879" width="86" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="458" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">mask_shift</text>
  <rect x="501" y="879" width="135" height="12" fill="#4ECDC455" stroke="#bbb" stroke-width="0.5"/>
  <text x="568" y="888" text-anchor="middle" font-size="7.5" font-family="Arial">mask_offset</text>
  <text x="28" y="902" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="893" width="109" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="188" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="243" y="893" width="86" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="286" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="329" y="893" width="86" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="372" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="415" y="893" width="86" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="458" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="501" y="893" width="135" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="568" y="902" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="330" y="960" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">ACC  (17 bits = 4b opcode + 4+3+4+2b fields)</text>
  <rect x="24" y="968" width="144" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="96" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="96" y="1010" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="168" y="968" width="144" height="64" fill="#45B7D1" stroke="#333" stroke-width="1" rx="3"/>
  <text x="240" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">DstructureCrIdx</text>
  <text x="240" y="1010" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="312" y="968" width="108" height="64" fill="#45B7D1" stroke="#333" stroke-width="1" rx="3"/>
  <text x="366" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrdIdx</text>
  <text x="366" y="1010" text-anchor="middle" font-size="8" font-family="Arial">3b</text>
  <rect x="420" y="968" width="144" height="64" fill="#45B7D1" stroke="#333" stroke-width="1" rx="3"/>
  <text x="492" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="492" y="1010" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="564" y="968" width="72" height="64" fill="#45B7D1" stroke="#333" stroke-width="1" rx="3"/>
  <text x="600" y="996" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">VerticalStride</text>
  <text x="600" y="1010" text-anchor="middle" font-size="8" font-family="Arial">2b</text>
  <text x="28" y="1048" font-size="8.5" font-family="monospace" fill="#222">ACC.ADD</text>
  <rect x="134" y="1039" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1048" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="252" y="1039" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1048" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1039" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1048" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="458" y="1039" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1048" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1062" font-size="8.5" font-family="monospace" fill="#222">ACC.ADD.FIRST</text>
  <rect x="134" y="1053" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1062" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="252" y="1053" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1062" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1053" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1062" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="458" y="1053" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1062" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1076" font-size="8.5" font-family="monospace" fill="#222">ACC.MAX</text>
  <rect x="134" y="1067" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1076" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="252" y="1067" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1076" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1067" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1076" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="458" y="1067" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1076" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1090" font-size="8.5" font-family="monospace" fill="#222">ACC.MAX.FIRST</text>
  <rect x="134" y="1081" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1090" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="252" y="1081" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1090" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1081" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1090" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="458" y="1081" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1090" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1104" font-size="8.5" font-family="monospace" fill="#222">ACC.RESHAPE</text>
  <rect x="134" y="1095" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1104" text-anchor="middle" font-size="7.5" font-family="Arial">reshape_mask:LrOrReshapeMaskImmediate</text>
  <rect x="252" y="1095" width="88" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1104" text-anchor="middle" font-size="7.5" font-family="Arial">source</text>
  <rect x="340" y="1095" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1104" text-anchor="middle" font-size="7.5" font-family="Arial">dest:LrdIdx</text>
  <rect x="458" y="1095" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1104" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1118" font-size="8.5" font-family="monospace" fill="#222">ACC.STRIDE</text>
  <rect x="134" y="1109" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1118" text-anchor="middle" font-size="7.5" font-family="Arial">elements_in_row:ElementsInRow</text>
  <rect x="252" y="1109" width="88" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1118" text-anchor="middle" font-size="7.5" font-family="Arial">horizontal_stride:HorizontalStride</text>
  <rect x="340" y="1109" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1118" text-anchor="middle" font-size="7.5" font-family="Arial">offset</text>
  <rect x="458" y="1109" width="178" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1118" text-anchor="middle" font-size="7.5" font-family="Arial">vertical_stride</text>
  <text x="28" y="1132" font-size="8.5" font-family="monospace" fill="#222">ACC.SUB</text>
  <rect x="134" y="1123" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1132" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="252" y="1123" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1132" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1123" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1132" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="458" y="1123" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1132" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1146" font-size="8.5" font-family="monospace" fill="#222">ACC.SUB.FIRST</text>
  <rect x="134" y="1137" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1146" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="252" y="1137" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1146" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1137" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1146" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="458" y="1137" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1146" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1160" font-size="8.5" font-family="monospace" fill="#222">AGG.MAX</text>
  <rect x="134" y="1151" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1160" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx</text>
  <rect x="252" y="1151" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1160" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1151" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1160" text-anchor="middle" font-size="7.5" font-family="Arial">dest_slot</text>
  <rect x="458" y="1151" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1160" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1174" font-size="8.5" font-family="monospace" fill="#222">AGG.MAX.FIRST</text>
  <rect x="134" y="1165" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1174" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx</text>
  <rect x="252" y="1165" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1174" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1165" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1174" text-anchor="middle" font-size="7.5" font-family="Arial">dest_slot</text>
  <rect x="458" y="1165" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1174" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1188" font-size="8.5" font-family="monospace" fill="#222">AGG.SUM</text>
  <rect x="134" y="1179" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1188" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx</text>
  <rect x="252" y="1179" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1188" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1179" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1188" text-anchor="middle" font-size="7.5" font-family="Arial">dest_slot</text>
  <rect x="458" y="1179" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1188" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1202" font-size="8.5" font-family="monospace" fill="#222">AGG.SUM.FIRST</text>
  <rect x="134" y="1193" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1202" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx</text>
  <rect x="252" y="1193" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1202" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1193" width="118" height="12" fill="#45B7D155" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1202" text-anchor="middle" font-size="7.5" font-family="Arial">dest_slot</text>
  <rect x="458" y="1193" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1202" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1216" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="1207" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="193" y="1216" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="252" y="1207" width="88" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="296" y="1216" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="340" y="1207" width="118" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="399" y="1216" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="458" y="1207" width="178" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="547" y="1216" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="330" y="1274" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">AAQ  (9 bits = 1b opcode + 4+4b fields)</text>
  <rect x="24" y="1282" width="68" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="58" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="58" y="1324" text-anchor="middle" font-size="8" font-family="Arial">1b</text>
  <rect x="92" y="1282" width="272" height="64" fill="#9B59B6" stroke="#333" stroke-width="1" rx="3"/>
  <text x="228" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">ActivationFn</text>
  <text x="228" y="1324" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="364" y="1282" width="272" height="64" fill="#9B59B6" stroke="#333" stroke-width="1" rx="3"/>
  <text x="500" y="1310" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">DstructureCrIdx</text>
  <text x="500" y="1324" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <text x="28" y="1362" font-size="8.5" font-family="monospace" fill="#222">ACTIVATE.QUANTIZE</text>
  <rect x="134" y="1353" width="223" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="245" y="1362" text-anchor="middle" font-size="7.5" font-family="Arial">activation_fn</text>
  <rect x="357" y="1353" width="279" height="12" fill="#9B59B655" stroke="#bbb" stroke-width="0.5"/>
  <text x="496" y="1362" text-anchor="middle" font-size="7.5" font-family="Arial">cr_idx</text>
  <text x="28" y="1376" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="1367" width="223" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="245" y="1376" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="357" y="1367" width="279" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="496" y="1376" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="330" y="1434" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">STORE  (9 bits = 1b opcode + 4+4b fields)</text>
  <rect x="24" y="1442" width="68" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="58" y="1470" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="58" y="1484" text-anchor="middle" font-size="8" font-family="Arial">1b</text>
  <rect x="92" y="1442" width="272" height="64" fill="#E74C3C" stroke="#333" stroke-width="1" rx="3"/>
  <text x="228" y="1470" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">CrIdx</text>
  <text x="228" y="1484" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="364" y="1442" width="272" height="64" fill="#E74C3C" stroke="#333" stroke-width="1" rx="3"/>
  <text x="500" y="1470" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="500" y="1484" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <text x="28" y="1522" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="1513" width="223" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="245" y="1522" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="357" y="1513" width="279" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="496" y="1522" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="1536" font-size="8.5" font-family="monospace" fill="#222">ST.AAQ</text>
  <rect x="134" y="1527" width="223" height="12" fill="#E74C3C55" stroke="#bbb" stroke-width="0.5"/>
  <text x="245" y="1536" text-anchor="middle" font-size="7.5" font-family="Arial">base</text>
  <rect x="357" y="1527" width="279" height="12" fill="#E74C3C55" stroke="#bbb" stroke-width="0.5"/>
  <text x="496" y="1536" text-anchor="middle" font-size="7.5" font-family="Arial">offset</text>
</svg>

### Emulator word (186 bits)

The word the emulator decodes and `.bin` files hold: the hardware word followed by
the emulator-only slots (`ACC_STORE`, `BREAK`) below its bit 0. Its top
155 bits are the hardware word.

#### Emulator-only slots: union layout

<svg xmlns="http://www.w3.org/2000/svg" width="660" height="410" viewBox="0 0 660 410">
  <rect x="0" y="0" width="660" height="410" fill="white"/>
  <text x="330" y="38" text-anchor="middle" font-size="14" font-weight="bold" font-family="Arial">IPU VLIW — Union Field Layout (per slot, MSB → LSB)</text>
  <text x="330" y="68" text-anchor="middle" font-size="11" font-weight="bold" font-family="Arial">ACC_STORE  (9 bits = 1b opcode + 4+4b fields)</text>
  <rect x="24" y="76" width="68" height="64" fill="#D3D3D3" stroke="#333" stroke-width="1" rx="3"/>
  <text x="58" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">opcode</text>
  <text x="58" y="118" text-anchor="middle" font-size="8" font-family="Arial">1b</text>
  <rect x="92" y="76" width="272" height="64" fill="#C0392B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="228" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">CrIdx</text>
  <text x="228" y="118" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <rect x="364" y="76" width="272" height="64" fill="#C0392B" stroke="#333" stroke-width="1" rx="3"/>
  <text x="500" y="104" text-anchor="middle" font-size="9" font-weight="bold" font-family="Arial">LrIdx</text>
  <text x="500" y="118" text-anchor="middle" font-size="8" font-family="Arial">4b</text>
  <text x="28" y="156" font-size="8.5" font-family="monospace" fill="#222">NOP</text>
  <rect x="134" y="147" width="223" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="245" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <rect x="357" y="147" width="279" height="12" fill="#F8F8F8" stroke="#bbb" stroke-width="0.5"/>
  <text x="496" y="156" text-anchor="middle" font-size="7.5" font-family="Arial">—</text>
  <text x="28" y="170" font-size="8.5" font-family="monospace" fill="#222">ST.ACC</text>
  <rect x="134" y="161" width="223" height="12" fill="#C0392B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="245" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">base</text>
  <rect x="357" y="161" width="279" height="12" fill="#C0392B55" stroke="#bbb" stroke-width="0.5"/>
  <text x="496" y="170" text-anchor="middle" font-size="7.5" font-family="Arial">offset</text>
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

<svg width="888" height="751" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .inst-title { font-size: 14px; font-weight: bold; font-family: Arial; }
      .inst-label { font-size: 8px; font-weight: bold; font-family: Arial; }
      .field-label { font-size: 7px; font-family: Arial; }
      .bit-range { font-size: 8px; font-family: Arial; }
    </style>
  </defs>
  <rect x="0" y="0" width="888" height="751" fill="white" stroke="none"/>
  <text x="444.0" y="34" text-anchor="middle" class="inst-title">Emulator word Layout - 186 total bits</text>
  <rect x="80" y="55.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="99.0" text-anchor="end" class="bit-range">[185:160]</text>
  <text x="236.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">185</text>
  <text x="260.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">184</text>
  <text x="284.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">183</text>
  <text x="308.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">182</text>
  <text x="332.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">181</text>
  <text x="356.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">180</text>
  <text x="380.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">179</text>
  <text x="404.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">178</text>
  <text x="428.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">177</text>
  <text x="452.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">176</text>
  <text x="476.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">175</text>
  <text x="500.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">174</text>
  <text x="524.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">173</text>
  <text x="548.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">172</text>
  <text x="572.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">171</text>
  <text x="596.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">170</text>
  <text x="620.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">169</text>
  <text x="644.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">168</text>
  <text x="668.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">167</text>
  <text x="692.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">166</text>
  <text x="716.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">165</text>
  <text x="740.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">164</text>
  <text x="764.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">163</text>
  <text x="788.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">162</text>
  <text x="812.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">161</text>
  <text x="836.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">160</text>
  <rect x="776" y="55.0" width="72" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="812.0" y="78.5" text-anchor="middle" class="field-label">lr</text>
  <text x="812.0" y="89.5" text-anchor="middle" class="field-label">inst</text>
  <text x="812.0" y="100.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="812.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[162:159]</text>
  <rect x="656" y="55.0" width="120" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="716.0" y="84.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="716.0" y="95.0" text-anchor="middle" class="field-label">reg</text>
  <text x="716.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[167:163]</text>
  <rect x="536" y="55.0" width="120" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="596.0" y="84.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="596.0" y="95.0" text-anchor="middle" class="field-label">reg</text>
  <text x="596.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[172:168]</text>
  <rect x="296" y="55.0" width="240" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="416.0" y="84.0" text-anchor="middle" class="field-label">label</text>
  <text x="416.0" y="95.0" text-anchor="middle" class="field-label">token</text>
  <text x="416.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[182:173]</text>
  <rect x="224" y="55.0" width="72" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="260.0" y="78.5" text-anchor="middle" class="field-label">cond</text>
  <text x="260.0" y="89.5" text-anchor="middle" class="field-label">inst</text>
  <text x="260.0" y="100.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="260.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[185:183]</text>
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
  <rect x="776" y="135.0" width="72" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="812.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="812.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="812.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[130:127]</text>
  <rect x="584" y="135.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="680.0" y="164.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="680.0" y="175.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="680.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[138:131]</text>
  <rect x="488" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="536.0" y="158.5" text-anchor="middle" class="field-label">lr</text>
  <text x="536.0" y="169.5" text-anchor="middle" class="field-label">inst</text>
  <text x="536.0" y="180.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="536.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[142:139]</text>
  <rect x="392" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="440.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="440.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="440.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[146:143]</text>
  <rect x="296" y="135.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="344.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="344.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="344.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[150:147]</text>
  <rect x="104" y="135.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="200.0" y="164.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="200.0" y="175.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="200.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[158:151]</text>
  <rect x="80" y="135.0" width="24" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="158.5" text-anchor="middle" class="field-label">lr</text>
  <text x="92.0" y="169.5" text-anchor="middle" class="field-label">inst</text>
  <text x="92.0" y="180.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="92.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[162:159]</text>
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
  <rect x="824" y="215.0" width="24" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="836.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="836.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="836.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[96:93]</text>
  <rect x="728" y="215.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="776.0" y="244.0" text-anchor="middle" class="field-label">cr</text>
  <text x="776.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="776.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[100:97]</text>
  <rect x="680" y="215.0" width="48" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="704.0" y="238.5" text-anchor="middle" class="field-label">load</text>
  <text x="704.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="704.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="704.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[102:101]</text>
  <rect x="584" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="632.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="632.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="632.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[106:103]</text>
  <rect x="488" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="536.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="536.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="536.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[110:107]</text>
  <rect x="296" y="215.0" width="192" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="392.0" y="244.0" text-anchor="middle" class="field-label">addbi</text>
  <text x="392.0" y="255.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="392.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[118:111]</text>
  <rect x="200" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="248.0" y="238.5" text-anchor="middle" class="field-label">lr</text>
  <text x="248.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="248.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="248.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[122:119]</text>
  <rect x="104" y="215.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="152.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="152.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="152.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[126:123]</text>
  <rect x="80" y="215.0" width="24" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="92.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="92.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[130:127]</text>
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
  <rect x="800" y="295.0" width="48" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="824.0" y="318.5" text-anchor="middle" class="field-label">acc</text>
  <text x="824.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="824.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="824.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[65:62]</text>
  <rect x="728" y="295.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="764.0" y="313.0" text-anchor="middle" class="field-label">mult</text>
  <text x="764.0" y="324.0" text-anchor="middle" class="field-label">mask</text>
  <text x="764.0" y="335.0" text-anchor="middle" class="field-label">offset</text>
  <text x="764.0" y="346.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="764.0" y="360.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[68:66]</text>
  <rect x="632" y="295.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="680.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="680.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="680.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[72:69]</text>
  <rect x="536" y="295.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="584.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="584.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="584.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[76:73]</text>
  <rect x="440" y="295.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="488.0" y="318.5" text-anchor="middle" class="field-label">dstructure</text>
  <text x="488.0" y="329.5" text-anchor="middle" class="field-label">cr</text>
  <text x="488.0" y="340.5" text-anchor="middle" class="field-label">reg</text>
  <text x="488.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[80:77]</text>
  <rect x="320" y="295.0" width="120" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="380.0" y="324.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="380.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="380.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[85:81]</text>
  <rect x="248" y="295.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="284.0" y="318.5" text-anchor="middle" class="field-label">mult</text>
  <text x="284.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="284.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="284.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[88:86]</text>
  <rect x="152" y="295.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="200.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="200.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="200.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[92:89]</text>
  <rect x="80" y="295.0" width="72" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="116.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="116.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="116.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[96:93]</text>
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
  <rect x="776" y="375.0" width="72" height="80" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="812.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="812.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="812.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[34:31]</text>
  <rect x="680" y="375.0" width="96" height="80" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="728.0" y="404.0" text-anchor="middle" class="field-label">cr</text>
  <text x="728.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="728.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[38:35]</text>
  <rect x="656" y="375.0" width="24" height="80" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="668.0" y="398.5" text-anchor="middle" class="field-label">store</text>
  <text x="668.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="668.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="668.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[39:39]</text>
  <rect x="560" y="375.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="608.0" y="398.5" text-anchor="middle" class="field-label">dstructure</text>
  <text x="608.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="608.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="608.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[43:40]</text>
  <rect x="464" y="375.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="512.0" y="404.0" text-anchor="middle" class="field-label">activation</text>
  <text x="512.0" y="415.0" text-anchor="middle" class="field-label">fn</text>
  <text x="512.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[47:44]</text>
  <rect x="440" y="375.0" width="24" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="452.0" y="398.5" text-anchor="middle" class="field-label">aaq</text>
  <text x="452.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="452.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="452.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[48:48]</text>
  <rect x="392" y="375.0" width="48" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="416.0" y="404.0" text-anchor="middle" class="field-label">vertical</text>
  <text x="416.0" y="415.0" text-anchor="middle" class="field-label">stride</text>
  <text x="416.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[50:49]</text>
  <rect x="296" y="375.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="344.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="344.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="344.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[54:51]</text>
  <rect x="224" y="375.0" width="72" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="260.0" y="404.0" text-anchor="middle" class="field-label">lrd</text>
  <text x="260.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="260.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[57:55]</text>
  <rect x="128" y="375.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="176.0" y="398.5" text-anchor="middle" class="field-label">dstructure</text>
  <text x="176.0" y="409.5" text-anchor="middle" class="field-label">cr</text>
  <text x="176.0" y="420.5" text-anchor="middle" class="field-label">reg</text>
  <text x="176.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[61:58]</text>
  <rect x="80" y="375.0" width="48" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="104.0" y="398.5" text-anchor="middle" class="field-label">acc</text>
  <text x="104.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="104.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="104.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[65:62]</text>
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
  <rect x="752" y="455.0" width="96" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="800.0" y="484.0" text-anchor="middle" class="field-label">lr</text>
  <text x="800.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="800.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[3:0]</text>
  <rect x="368" y="455.0" width="384" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="560.0" y="484.0" text-anchor="middle" class="field-label">break</text>
  <text x="560.0" y="495.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="560.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[19:4]</text>
  <rect x="320" y="455.0" width="48" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="344.0" y="478.5" text-anchor="middle" class="field-label">break</text>
  <text x="344.0" y="489.5" text-anchor="middle" class="field-label">inst</text>
  <text x="344.0" y="500.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="344.0" y="514.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[21:20]</text>
  <rect x="224" y="455.0" width="96" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="272.0" y="484.0" text-anchor="middle" class="field-label">lr</text>
  <text x="272.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="272.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[25:22]</text>
  <rect x="128" y="455.0" width="96" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="176.0" y="484.0" text-anchor="middle" class="field-label">cr</text>
  <text x="176.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="176.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[29:26]</text>
  <rect x="104" y="455.0" width="24" height="80" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="116.0" y="473.0" text-anchor="middle" class="field-label">acc</text>
  <text x="116.0" y="484.0" text-anchor="middle" class="field-label">store</text>
  <text x="116.0" y="495.0" text-anchor="middle" class="field-label">inst</text>
  <text x="116.0" y="506.0" text-anchor="middle" class="field-label">opcode</text>
  <text x="116.0" y="520.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[30:30]</text>
  <rect x="80" y="455.0" width="24" height="80" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="484.0" text-anchor="middle" class="field-label">lr</text>
  <text x="92.0" y="495.0" text-anchor="middle" class="field-label">reg</text>
  <text x="92.0" y="509.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[34:31]</text>
  <text x="80" y="555.0" class="inst-label" style="font-weight: bold;">Instruction Type Colors:</text>
  <rect x="80" y="560.0" width="12" height="12" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="572.0" class="field-label" style="font-size: 9px;">BreakInst (Break / Debug)</text>
  <rect x="80" y="575.0" width="12" height="12" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="587.0" class="field-label" style="font-size: 9px;">LoadInst (Memory Load)</text>
  <rect x="80" y="590.0" width="12" height="12" fill="#E74C3C" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="602.0" class="field-label" style="font-size: 9px;">StoreInst (Memory Store)</text>
  <rect x="80" y="605.0" width="12" height="12" fill="#C0392B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="617.0" class="field-label" style="font-size: 9px;">AccStoreInst (ST.ACC, simulation-only)</text>
  <rect x="80" y="620.0" width="12" height="12" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="632.0" class="field-label" style="font-size: 9px;">MultInst (Multiply)</text>
  <rect x="80" y="635.0" width="12" height="12" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="647.0" class="field-label" style="font-size: 9px;">AccInst (Accumulator)</text>
  <rect x="80" y="650.0" width="12" height="12" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="662.0" class="field-label" style="font-size: 9px;">AaqInst (Activation and Quantization)</text>
  <rect x="80" y="665.0" width="12" height="12" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="677.0" class="field-label" style="font-size: 9px;">LrInst (Link Register)</text>
  <rect x="80" y="680.0" width="12" height="12" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="692.0" class="field-label" style="font-size: 9px;">CondInst (Conditional)</text>
</svg>


---


## Conditional Branch Instructions

Control flow instructions for branching based on conditions or unconditionally.

### `BEQ` — Branch if Equal

**Syntax:** `BEQ reg1 reg2 label`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2`.

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR15) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR15) |
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

### `BNE` — Branch if Not Equal

**Syntax:** `BNE reg1 reg2 label`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2`.

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR15) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR15) |
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

### `BLT.S` — Branch if Less Than (signed)

**Syntax:** `BLT.S reg1 reg2 label`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2`.

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR15) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR15) |
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

### `BGE.S` — Branch if Greater or Equal (signed)

**Syntax:** `BGE.S reg1 reg2 label`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `reg1`, `CR` via `reg1`, `LR` via `reg2`, `CR` via `reg2`.

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR15) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR15) |
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

### `BR` — Branch Register

**Syntax:** `BR reg`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `reg`, `CR` via `reg`.

**Faults:** F-2 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LcrIdx`](operand-types.md#lcridx) | reg: Register containing target address (LR0–LR15 or CR0–CR15) |

**General description:**
Branch to address in register.

**Pseudo code:**
`PC = reg`


### `HALT` — Halt

**Syntax:** `HALT`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** no register.

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

**Reads:** at cycle start `CR` via `src`.

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LrIdx`](operand-types.md#lridx) | reg: Loop register (LR0–LR15) |
| `src` | [`CrIdx`](operand-types.md#cridx) | src: Source configuration register (CR0–CR15) |

**General description:**
Copy a 32-bit value from a configuration register into a loop register.

**Pseudo code:**
`reg = cr[src]`

**Example of usage:**
```asm
SET LR0 CR1;;
```

### `ADD` — Add

**Syntax:** `ADD dest src_a src_b`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `src_a`, `LR` via `src_b`, `CR` via `src_b`.

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination local register (LR0–LR15) |
| `src_a` | [`LrIdx`](operand-types.md#lridx) | src_a: First source local register (LR0–LR15) |
| `src_b` | [`LcrIdx`](operand-types.md#lcridx) | src_b: Second source — LR0–LR15 or CR0–CR15 |

**General description:**
Add two register sources and store the result in the destination LR.

**Pseudo code:**
`dest = src_a + src_b`

**Example of usage:**
```asm
ADD LR0 LR1 LR2;;
ADD LR3 LR1 CR5;;
```

### `SUB` — Subtract

**Syntax:** `SUB dest src_a src_b`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `src_a`, `LR` via `src_b`, `CR` via `src_b`.

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination local register (LR0–LR15) |
| `src_a` | [`LrIdx`](operand-types.md#lridx) | src_a: First source local register (LR0–LR15) |
| `src_b` | [`LcrIdx`](operand-types.md#lcridx) | src_b: Second source — LR0–LR15 or CR0–CR15 |

**General description:**
Subtract the second source from the first and store the result in the destination LR.

**Pseudo code:**
`dest = src_a - src_b`

**Example of usage:**
```asm
SUB LR0 LR1 LR2;;
SUB LR3 LR1 CR5;;
```

### `INC.MODP2` — Increment Loop Register Modulo Power of Two

**Syntax:** `INC.MODP2 dest step k`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `step`, `CR` via `step`, `LR` via `dest`.

**Faults:** F-1, F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination loop register (LR0–LR15); read and written |
| `step` | [`LcrIdx`](operand-types.md#lcridx) | step: Signed 32-bit increment from LR0–LR15 or CR0–CR15 |
| `k` | [`LrModPow2KImmediate`](operand-types.md#lrmodpow2kimmediate) | k: Immediate in [1, 9]; encoded in 4 bits as (k − 1); mask = (1 << k) - 1 |

**General description:**
Add a loop or configuration register into the destination loop register, then mask to k low bits (mod 2^k).

**Pseudo code:**
`dest <- (dest + step) & ((1 << k) - 1)`

**Example of usage:**
```asm
INC.MODP2 LR2 LR3 4;;
```

### `INC` — Increment

**Syntax:** `INC dest imm`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `dest`.

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

### `DEC` — Decrement

**Syntax:** `DEC dest imm`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `dest`.

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

### `ADDB` — Add Byte (Broadcast, Register)

**Syntax:** `ADDB dest src_b`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `src_b`, `CR` via `src_b`, `LR` via `dest`.

**Faults:** F-3 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrdIdx`](operand-types.md#lrdidx) | dest: Destination register pair (LRD0, LRD2, LRD4, ..., LRD14 — named after the lower register); LRDn = LR(n+1):LR(n), 8 byte elements; also the implicit source |
| `src_b` | [`LcrIdx`](operand-types.md#lcridx) | src_b: LR0–LR15 or CR0–CR15; low byte is read and reinterpreted as a signed two's-complement byte |

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

### `ADDBI` — Add Byte Immediate (Broadcast)

**Syntax:** `ADDBI dest imm`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `dest`.

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

---

## LOAD Instructions

First-stage memory loads that feed the multiply unit (mult-stage registers, cyclic register, and mask register).

### `LD.MULT` — Load Register

**Syntax:** `LD.MULT dest offset base`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `CR` via `base`; forwarded `LR` via `offset`.

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
`dest = Memory[row(offset + base)]  # one row = 128 elements (128 B narrow, 512 B wide-vector debug)`

**Example of usage:**
```asm
SET LR0 CR1;;
LD.MULT R0 LR0 CR0;;
```

### `LD.MULT.CYCLIC` — Load Cyclic Register

**Syntax:** `LD.MULT.CYCLIC offset base index`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `CR` via `base`; forwarded `LR` via `offset`, `LR` via `index`.

**Faults:** F-4, F-6, F-7 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15), in XMEM rows. |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR15), in XMEM rows. |
| `index` | [`LrIdx`](operand-types.md#lridx) | index: Index inside cyclic register (LR0–LR15); must hold 0, 128, 256, or 384 — the four R_CYCLIC slot boundaries, counted in **elements**. Any other value raises an error. All four are valid in **both** modes: R_CYCLIC is a 512-element ring in narrow and wide-vector debug mode alike, and a load always replaces one whole 128-element slot. |

**General description:**
Load with cyclic addressing into R_CYCLIC.

**Pseudo code:**
`R_CYCLIC[index .. index+127] = Memory[row(offset + base)]   # index ∈ {0, 128, 256, 384}`


### `LD.MASK` — Load Mask Register

**Syntax:** `LD.MASK offset base`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `CR` via `base`; forwarded `LR` via `offset`.

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

---

## MULT Instructions

Multiplication instructions for element-wise and element-vector operations.
The multiplication result (`mult_result`) is forwarded to the ACC stage in the CPU and not stored in any register in the way.

### `MULT.RC.VV` — RC Vector × Ra Vector Multiply

**Syntax:** `MULT.RC.VV rc_idx ra mask_offset mask_shift cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `R` via `ra`, `R_CYCLIC`, `CR` via `cr_idx`; forwarded `LR` via `rc_idx`, `LR` via `mask_shift`, `R_MASK`.

**Faults:** F-1 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](operand-types.md#lridx) | `rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`). |
| `ra` | [`MultStageReg`](operand-types.md#multstagereg) | `ra`: **`R0`** \| **`R1`** — multiplicand mult-stage register (same cycle as `LD.MULT` into **`R0`**/**`R1`** is allowed). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | `cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly). |

**General description:**
Multiply R_CYCLIC[rc_idx:rc_idx+128] by Ra (R0 or R1) element-wise. The partition used by mask_shift comes from cr_idx's dstructure configuration (any CR0-CR15; must be named explicitly).

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`For each element i: MULT_RES[i] = ipu_mult(R_CYCLIC[(rc_idx + i) % 512], ra[i]); then apply mask and shift using CR[cr_idx].partition.`

**Example of usage:**
```asm
MULT.RC.VV LR0 R0 0 LR2 CR15;;
```

### `MULT.RC.VE` — RC Vector × Scalar Multiply

**Syntax:** `MULT.RC.VE rc_idx src mask_offset mask_shift cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `CR` via `src`, `R`, `R_CYCLIC`, `CR` via `cr_idx`; forwarded `LR` via `rc_idx`, `LR` via `mask_shift`, `LR` via `src`, `R_MASK`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](operand-types.md#lridx) | `rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`). |
| `src` | [`LcrIdx`](operand-types.md#lcridx) | `src`: **`LR0`**…**`LR15`** \| **`CR0`**…**`CR15`** — if an LR, its stored value selects the scalar from R0/R1 (0..127 → `R0[idx]`, 128..255 → `R1[idx - 128]`); if a CR, it supplies the scalar directly ([CR scalars](../../explanation/element-formats.md#cr-scalars)). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | `cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly). |

**General description:**
Multiply R_CYCLIC[rc_idx:rc_idx+128] by a scalar, element-wise. The scalar is either a single element of R0/R1 (selected by an LR value) or a CR register value. The partition used by mask_shift comes from cr_idx's dstructure configuration (any CR0-CR15; must be named explicitly).

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`For each element i: MULT_RES[i] = ipu_mult(R_CYCLIC[(rc_idx + i) % 512], src_value); then apply mask and shift using CR[cr_idx].partition.`

**Example of usage:**
```asm
MULT.RC.VE LR0 LR3 0 LR2 CR15;;
```

### `MULT.RC.VS` — RC Vector Self-Multiply (Square)

**Syntax:** `MULT.RC.VS rc_idx mask_offset mask_shift cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `R_CYCLIC`, `CR` via `cr_idx`; forwarded `LR` via `rc_idx`, `LR` via `mask_shift`, `R_MASK`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `rc_idx` | [`LrIdx`](operand-types.md#lridx) | `rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | `cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly). |

**General description:**
Square R_CYCLIC[rc_idx:rc_idx+128] element-wise. The partition used by mask_shift comes from cr_idx's dstructure configuration (any CR0-CR15; must be named explicitly).

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`For each element i: rb = R_CYCLIC[(rc_idx + i) % 512]; MULT_RES[i] = ipu_mult(rb, rb); then apply mask and shift using CR[cr_idx].partition.`

**Example of usage:**
```asm
MULT.RC.VS LR0 0 LR2 CR15;;
```

### `MULT.VE` — Ra Vector × CR Scalar Multiply

**Syntax:** `MULT.VE ra_idx cr_idx mask_offset mask_shift dstructure_cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `CR` via `cr_idx`, `R`, `CR` via `dstructure_cr_idx`; forwarded `LR` via `ra_idx`, `LR` via `mask_shift`, `R_MASK`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `ra_idx` | [`LrIdx`](operand-types.md#lridx) | `ra_idx`: **`LR0`**…**`LR15`** — base element offset into combined Ra (`R0` ++ `R1`, 256 elements, cyclic mod 256). |
| `cr_idx` | [`CrIdx`](operand-types.md#cridx) | `cr_idx`: **`CR0`**…**`CR15`** — CR register that supplies the scalar multiplier ([CR scalars](../../explanation/element-formats.md#cr-scalars)). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `dstructure_cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | `dstructure_cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly). |

**General description:**
Multiply Ra[ra_idx:ra_idx+128] (combined R0/R1) by a CR scalar, element-wise. The partition used by mask_shift comes from dstructure_cr_idx's dstructure configuration (any CR0-CR15; must be named explicitly).

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`For each element i: ra = Ra[(ra_idx + i) % 256]; MULT_RES[i] = ipu_mult(ra, CR[cr_idx][0]); then apply mask and shift using CR[dstructure_cr_idx].partition.`

**Example of usage:**
```asm
MULT.VE LR0 CR3 0 LR2 CR15;;
```

### `MULT.EE` — Ra Element × CR Scalar Multiply (broadcast)

**Syntax:** `MULT.EE ra_idx cr_idx mask_offset mask_shift dstructure_cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `CR` via `cr_idx`, `R`, `CR` via `dstructure_cr_idx`; forwarded `LR` via `ra_idx`, `LR` via `mask_shift`, `R_MASK`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `ra_idx` | [`LrIdx`](operand-types.md#lridx) | `ra_idx`: **`LR0`**…**`LR15`** — index of the single Ra element to read (combined `R0` ++ `R1`, 256 elements, mod 256). |
| `cr_idx` | [`CrIdx`](operand-types.md#cridx) | `cr_idx`: **`CR0`**…**`CR15`** — CR register that supplies the scalar multiplier ([CR scalars](../../explanation/element-formats.md#cr-scalars)). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end). |
| `dstructure_cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | `dstructure_cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly). |

**General description:**
Multiply a single Ra element by a CR scalar; broadcast the product to all 128 elements. The partition used by mask_shift comes from dstructure_cr_idx's dstructure configuration (any CR0-CR15; must be named explicitly).

> Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.

**Pseudo code:**
`ra = Ra[ra_idx % 256]; result = ipu_mult(ra, CR[cr_idx][0]); for each element i: MULT_RES[i] = result; then apply mask and shift using CR[dstructure_cr_idx].partition.`

**Example of usage:**
```asm
MULT.EE LR0 CR3 0 LR2 CR15;;
```

---

## ACC Instructions

Accumulation instructions for combining values with optional masking and shifting.

### `ACC.ADD` — Accumulate Add

**Syntax:** `ACC.ADD`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `R_ACC`; forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Running add accumulation: add multiply result into each R_ACC element.

**Pseudo code:**
`R_ACC[i] += MULT_RES[i]  # for each element i`

**Example of usage:**
```asm
ACC.ADD;;
```

### `ACC.ADD.FIRST` — Accumulate Add (First)

**Syntax:** `ACC.ADD.FIRST`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Clean-init add: overwrite each R_ACC element with the multiply result (ignores previous R_ACC).

**Pseudo code:**
`R_ACC[i] = MULT_RES[i]  # for each element i`

**Example of usage:**
```asm
ACC.ADD.FIRST;;
```

### `ACC.MAX` — Accumulate Max

**Syntax:** `ACC.MAX`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `R_ACC`; forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Running max accumulation: each R_ACC element takes the max of its current value and the multiply result.

**Pseudo code:**
`R_ACC[i] = max(R_ACC[i], MULT_RES[i])  # for each element i`

**Example of usage:**
```asm
ACC.MAX;;
```

### `ACC.MAX.FIRST` — Accumulate Max (First)

**Syntax:** `ACC.MAX.FIRST`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Clean-init max: overwrite each R_ACC element unconditionally with the multiply result.

**Pseudo code:**
`R_ACC[i] = MULT_RES[i]  # for each element i (unconditional overwrite)`

**Example of usage:**
```asm
ACC.MAX.FIRST;;
```

### `ACC.SUB` — Accumulate Subtract

**Syntax:** `ACC.SUB`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `R_ACC`; forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Running subtract accumulation: subtract the multiply result from each R_ACC element each cycle.

**Pseudo code:**
`R_ACC[i] -= MULT_RES[i]  # for each element i`

**Example of usage:**
```asm
ACC.SUB;;
```

### `ACC.SUB.FIRST` — Accumulate Subtract (First)

**Syntax:** `ACC.SUB.FIRST`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Clean-init subtract: set each R_ACC element to the negated multiply result.

**Pseudo code:**
`R_ACC[i] = -MULT_RES[i]  # for each element i`

**Example of usage:**
```asm
ACC.SUB.FIRST;;
```

### `ACC.STRIDE` — Accumulator Stride

**Syntax:** `ACC.STRIDE elements_in_row horizontal_stride vertical_stride offset`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** forwarded `LR` via `offset`, `MULT_RES`.

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

### `AGG.SUM.FIRST` — Aggregate Sum (First)

**Syntax:** `AGG.SUM.FIRST dest_slot cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `dest_slot`, `CR` via `cr_idx`; forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](operand-types.md#lridx) | dest_slot: LR register whose value gives the destination slot in R_ACC (0–127) |
| `cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly) |

**General description:**
Sum active MULT_RES elements and write the result into R_ACC at the slot given by LR. The current value at the destination slot is NOT included in the sum (clean initialisation). Only the active element count (``valid_elements``) is read from ``cr_idx``'s dstructure configuration (any CR0-CR15; must be named explicitly).

**Pseudo code:**
`Let n = min(CR[cr_idx].valid_elements, 128). dest = LR[dest_slot] % 128. R_ACC[dest] = sum(MULT_RES[0..n-1]).`

**Example of usage:**
```asm
AGG.SUM.FIRST LR0 CR3;;
```

### `AGG.SUM` — Aggregate Sum

**Syntax:** `AGG.SUM dest_slot cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `dest_slot`, `CR` via `cr_idx`, `R_ACC`; forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](operand-types.md#lridx) | dest_slot: LR register whose value gives the destination slot in R_ACC (0–127) |
| `cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly) |

**General description:**
Sum active MULT_RES elements and ADD the result to R_ACC at the slot given by LR (running cross-cycle accumulation). Only the active element count (``valid_elements``) is read from ``cr_idx``'s dstructure configuration (any CR0-CR15; must be named explicitly).

**Pseudo code:**
`Let n = min(CR[cr_idx].valid_elements, 128). dest = LR[dest_slot] % 128. R_ACC[dest] = sum(MULT_RES[0..n-1]) + R_ACC[dest].`

**Example of usage:**
```asm
AGG.SUM LR0 CR3;;
```

### `AGG.MAX.FIRST` — Aggregate Max (First)

**Syntax:** `AGG.MAX.FIRST dest_slot cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `dest_slot`, `CR` via `cr_idx`; forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](operand-types.md#lridx) | dest_slot: LR register whose value gives the destination slot in R_ACC (0–127) |
| `cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly) |

**General description:**
Find the maximum of active MULT_RES elements and write it into R_ACC at the slot given by LR. The current value at the destination slot is NOT used as a seed (clean initialisation). Only the active element count (``valid_elements``) is read from ``cr_idx``'s dstructure configuration (any CR0-CR15; must be named explicitly).

**Pseudo code:**
`Let n = min(CR[cr_idx].valid_elements, 128). dest = LR[dest_slot] % 128. R_ACC[dest] = max(MULT_RES[0..n-1]); when n = 0 the identity seed (INT32_MIN for integer elements, -inf for float elements) is written.`

**Example of usage:**
```asm
AGG.MAX.FIRST LR0 CR3;;
```

### `AGG.MAX` — Aggregate Max

**Syntax:** `AGG.MAX dest_slot cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `dest_slot`, `CR` via `cr_idx`, `R_ACC`; forwarded `MULT_RES`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest_slot` | [`LrIdx`](operand-types.md#lridx) | dest_slot: LR register whose value gives the destination slot in R_ACC (0–127) |
| `cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly) |

**General description:**
Find the maximum of active MULT_RES elements seeded with the current destination slot value (running cross-cycle max). Only the active element count (``valid_elements``) is read from ``cr_idx``'s dstructure configuration (any CR0-CR15; must be named explicitly).

**Pseudo code:**
`Let n = min(CR[cr_idx].valid_elements, 128). dest = LR[dest_slot] % 128. R_ACC[dest] = max(MULT_RES[0..n-1], R_ACC[dest]).`

**Example of usage:**
```asm
AGG.MAX LR0 CR3;;
```

### `ACC.RESHAPE` — Reshape

**Syntax:** `ACC.RESHAPE source dest reshape_mask`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `LR` via `source`, `LR` via `dest`, `MULT_RES`; forwarded `LR` via `reshape_mask`.

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
`mult_res = MULT_RES (cycle-start value)
mask = reshape_mask if immediate else LR[reshape_mask]
if mask > 8: raise error
for i in mask..7:
    if source[i] >= 128 or dest[i] >= 128: raise error
    R_ACC[dest[i]] = mult_res[source[i]]`

**Example of usage:**
```asm
ACC.RESHAPE LRD0 LRD2 0;;
```

---

## AAQ Instructions

Activation and quantization: apply activation to R_ACC elements and write quantized INT8 bytes to POST_AAQ_REG.

### `ACTIVATE.QUANTIZE` — Activate and Quantize

**Syntax:** `ACTIVATE.QUANTIZE activation_fn cr_idx`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `CR` via `cr_idx`; forwarded `R_ACC`.

**Faults:** F-1 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `activation_fn` | [`ActivationFn`](operand-types.md#activationfn) | activation_fn: keyword naming the activation (one of identity, relu, relu6, sigmoid, tanh, gelu, softplus, elu, exp2, reciprocal, rsqrt, silu, window; see ACTIVATION_FN_NAMES) |
| `cr_idx` | [`DstructureCrIdx`](operand-types.md#dstructurecridx) | cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly) |

**General description:**
Read active elements from the forwarded R_ACC (this cycle's ACC result), apply the selected element-wise activation, clamp the results to the INT8 range ``[-128, 127]``, and write the resulting bytes to the leading active-element positions of **``POST_AAQ_REG``**; all remaining bytes are zeroed. ``r_acc`` is not modified. Requires INT8 mode. The active element count is taken from ``cr_idx``'s dstructure ``valid_elements`` field (any CR0–CR15; must be named explicitly). Available activation functions: ``identity`` (0), ``relu`` (1), ``relu6`` (2), ``sigmoid`` (3), ``tanh`` (4), ``gelu`` (5), ``softplus`` (6), ``elu`` (7), ``exp2`` (8), ``reciprocal`` (9), ``rsqrt`` (10), ``silu`` (11), ``window`` (12).

**Pseudo code:**
`Requires the INT8 element format (IpuState.fmt == ElementFormat.INT8). Let n = min(CR[cr_idx].valid_elements, 128) and k = encoded activation index. For i in [0, n): POST_AAQ_REG[i] = clamp(round(activation_k(R_ACC[i])), -128, 127). POST_AAQ_REG[n..511] = 0. R_ACC is not modified. Clamping is a placeholder until a full per-element requantize is implemented. The selector uses four bits; encodings outside the thirteen named activations behave as identity. window(x) = 1 for a <= x < b and 0 otherwise. α for elu and the window bounds a, b are not ISA operands; see docs/content/building-applications.md#activations-emulator.`

**Example of usage:**
```asm
ACTIVATE.QUANTIZE relu CR15;;
```

---

## STORE Instructions

Last-stage memory stores that drain **POST_AAQ_REG** to external memory after activation and quantization.

### `ST.AAQ` — Store Post-AAQ register

**Syntax:** `ST.AAQ offset base`

**Tag:** `HARDWARE` (implemented by the IPU)

**Reads:** at cycle start `CR` via `base`; forwarded `LR` via `offset`, `POST_AAQ_REG`.

**Faults:** F-6, F-7 ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15), in XMEM rows. |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR15), in XMEM rows. |

**General description:**
Write the whole **`POST_AAQ_REG`** to one XMEM row. **Interim:** after `ACTIVATE.QUANTIZE` it holds 128 INT8 results followed by zeros, until quantization export is finalized.

**Pseudo code:**
`Memory[row(offset + base)] = POST_AAQ_REG (the whole register, one row, in every element format)`

**Example of usage:**
```asm
ST.AAQ LR0 CR0;;
```

---

## ACC_STORE Instructions

> **Emulator-only slot** — not implemented in real IPU hardware: not in the hardware word or the SV package.

Simulation-only slot for storing **R_ACC** to external memory (`ST.ACC`). Not implemented in real IPU hardware.

### `ST.ACC` — Store Accumulator

**Syntax:** `ST.ACC offset base`

**Tag:** `EMULATOR` (emulator only; not in the hardware word or the SV package)

**Reads:** at cycle start `CR` via `base`; forwarded `LR` via `offset`, `R_ACC`.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15), in XMEM rows. |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR15), in XMEM rows. |

**General description:**
Store **R_ACC** to external memory. **Simulation-only** — not implemented in real IPU hardware.

> This instruction lives in the simulation-only **acc_store** slot.

**Pseudo code:**
`Memory[row(offset + base)] = R_ACC (128 32-bit elements, in every element format)`

**Example of usage:**
```asm
ST.ACC LR0 CR1;;
```

---

## Break Instructions

> **Emulator-only slot** — not implemented in real IPU hardware: not in the hardware word or the SV package.

Debug break instructions for halting execution and entering debug mode.

### `BREAK` — Break

**Syntax:** `BREAK`

**Tag:** `EMULATOR` (emulator only; not in the hardware word or the SV package)

**Reads:** no register.

**Faults:** none ([fault list](../../requirements/faults.md#fault-list)).

**General description:**
Unconditional break.

**Pseudo code:**
`Halt execution`


### `BREAK.IFEQ` — Break if Equal

**Syntax:** `BREAK.IFEQ reg value`

**Tag:** `EMULATOR` (emulator only; not in the hardware word or the SV package)

**Reads:** at cycle start `LR` via `reg`.

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

---
