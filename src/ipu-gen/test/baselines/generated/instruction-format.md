# Instruction format

This page describes the **binary layout** of a single IPU **compound (VLIW) instruction**.
Field names, bit widths, opcode values, and union-field sharing are **generated** from
`instruction_spec.py` in `ipu_isa` (the same source the assembler and emulator use).

## Hardware word (169 bits)

The part of the instruction word the IPU decodes: the hardware slots
(`COND`, `LR`×3, `LOAD`, `MULT`, `ACC`, `AAQ`). The SystemVerilog package describes this
word (`IPU_COMPOUND_INST_WIDTH = 169`).

### Whole-word bit layout

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

### Per-slot union layout

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

## Emulator word (209 bits)

The word the emulator decodes and `.bin` files hold: the hardware word followed by
the emulator-only slots (`AAQ_EXACT`, `BREAK`) below its bit 0. Its top
169 bits are the hardware word.

### Emulator-only slots: union layout

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

### Whole-word bit layout

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

## Generated artifacts

These files are produced at documentation build time (and via the `ipu-gen` CLI). They are
**not** checked into the repository.

| Artifact | Description | Download |
|----------|-------------|----------|
| SystemVerilog package | `ipu_instr_pkg` — enums, per-slot structs (opcode + operand union), per-instruction `union packed` views, the hardware word as the compound type | [ipu_instr_pkg.sv](../../assets/ipu_instr_pkg.sv) |

### Regenerate locally

```bash
bazel build //docs:generate_instruction_format_artifacts
# outputs under bazel-bin/docs/

bazel run //src/ipu-gen:ipu-gen -- sv-package --output /tmp/ipu_instr_pkg.sv
```

## Related documentation

- [Assembly syntax](assembly-syntax.md) — how to write compound instructions in assembly
- [Operand types](operand-types.md) — semantic types of instruction operands
- [Instruction reference](instructions.md) — per-opcode documentation
