# Instruction format

This page describes the **binary layout** of a single IPU **compound (VLIW) instruction**.
Field names, bit widths, opcode values, and union-field sharing are **generated** from
`instruction_spec.py` in `ipu_isa` (the same source the assembler and emulator use).

## Hardware word (155 bits)

The part of the instruction word the IPU decodes: the hardware slots
(`COND`, `LR`×3, `LOAD`, `MULT`, `ACC`, `AAQ`, `STORE`). The SystemVerilog package describes this
word (`IPU_COMPOUND_INST_WIDTH = 155`).

### Whole-word bit layout

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

### Per-slot union layout

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

## Emulator word (186 bits)

The word the emulator decodes and `.bin` files hold: the hardware word followed by
the emulator-only slots (`ACC_STORE`, `BREAK`) below its bit 0. Its top
155 bits are the hardware word.

### Emulator-only slots: union layout

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

### Whole-word bit layout

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

## Generated artifacts

These files are produced at documentation build time (and via the `ipu-gen` CLI). They are
**not** checked into the repository.

| Artifact | Description | Download |
|----------|-------------|----------|
| SystemVerilog package | `ipu_instr_pkg` — enums, per-slot structs (opcode + operand union), per-instruction `union packed` views, the hardware word as the compound type | [ipu_instr_pkg.sv](assets/ipu_instr_pkg.sv) |

### Regenerate locally

```bash
bazel build //docs:generate_instruction_format_artifacts
# outputs under bazel-bin/docs/

bazel run //src/tools/ipu-gen:ipu-gen -- sv-package --output /tmp/ipu_instr_pkg.sv
```

## Related documentation

- [Assembly syntax](assembly-syntax.md) — how to write compound instructions in assembly
- [Operand types](operand-types.md) — semantic types of instruction operands
- [Instruction reference](instructions.md) — per-opcode documentation
