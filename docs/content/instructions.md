# IPU Assembly Instruction Reference

Per-opcode documentation is generated from `InstructionDoc` entries in `instruction_spec.py`. Operand **types** link to the shared [operand type reference](operand-types.md).

## Compound Instruction Layout

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
  <text x="444.0" y="34" text-anchor="middle" class="inst-title">CompoundInst Layout - 156 total bits</text>
  <rect x="80" y="55.0" width="768" height="80" fill="white" stroke="black" stroke-width="2"/>
  <text x="50" y="99.0" text-anchor="end" class="bit-range">[155:128]</text>
  <text x="188.0" y="67.0" text-anchor="middle" class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">155</text>
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
  <rect x="776" y="55.0" width="72" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="812.0" y="84.0" text-anchor="middle" class="field-label">cr</text>
  <text x="812.0" y="95.0" text-anchor="middle" class="field-label">reg</text>
  <text x="812.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[130:127]</text>
  <rect x="704" y="55.0" width="72" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="740.0" y="78.5" text-anchor="middle" class="field-label">xmem</text>
  <text x="740.0" y="89.5" text-anchor="middle" class="field-label">inst</text>
  <text x="740.0" y="100.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="740.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[133:131]</text>
  <rect x="608" y="55.0" width="96" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="656.0" y="84.0" text-anchor="middle" class="field-label">lr</text>
  <text x="656.0" y="95.0" text-anchor="middle" class="field-label">reg</text>
  <text x="656.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[137:134]</text>
  <rect x="224" y="55.0" width="384" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="416.0" y="84.0" text-anchor="middle" class="field-label">break</text>
  <text x="416.0" y="95.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="416.0" y="109.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[153:138]</text>
  <rect x="176" y="55.0" width="48" height="80" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="200.0" y="78.5" text-anchor="middle" class="field-label">break</text>
  <text x="200.0" y="89.5" text-anchor="middle" class="field-label">inst</text>
  <text x="200.0" y="100.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="200.0" y="114.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[155:154]</text>
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
  <rect x="824" y="135.0" width="24" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="836.0" y="164.0" text-anchor="middle" class="field-label">aaq</text>
  <text x="836.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="836.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[96:95]</text>
  <rect x="728" y="135.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="776.0" y="158.5" text-anchor="middle" class="field-label">acc</text>
  <text x="776.0" y="169.5" text-anchor="middle" class="field-label">inst</text>
  <text x="776.0" y="180.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="776.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[100:97]</text>
  <rect x="656" y="135.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="692.0" y="153.0" text-anchor="middle" class="field-label">mult</text>
  <text x="692.0" y="164.0" text-anchor="middle" class="field-label">mask</text>
  <text x="692.0" y="175.0" text-anchor="middle" class="field-label">offset</text>
  <text x="692.0" y="186.0" text-anchor="middle" class="field-label">immediate</text>
  <text x="692.0" y="200.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[103:101]</text>
  <rect x="560" y="135.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="608.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="608.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="608.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[107:104]</text>
  <rect x="464" y="135.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="512.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="512.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="512.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[111:108]</text>
  <rect x="368" y="135.0" width="96" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="416.0" y="164.0" text-anchor="middle" class="field-label">cr</text>
  <text x="416.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="416.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[115:112]</text>
  <rect x="296" y="135.0" width="72" height="80" fill="#4ECDC4" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="332.0" y="158.5" text-anchor="middle" class="field-label">mult</text>
  <text x="332.0" y="169.5" text-anchor="middle" class="field-label">inst</text>
  <text x="332.0" y="180.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="332.0" y="194.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[118:116]</text>
  <rect x="200" y="135.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="248.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="248.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="248.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[122:119]</text>
  <rect x="104" y="135.0" width="96" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="152.0" y="164.0" text-anchor="middle" class="field-label">lr</text>
  <text x="152.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="152.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[126:123]</text>
  <rect x="80" y="135.0" width="24" height="80" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="164.0" text-anchor="middle" class="field-label">cr</text>
  <text x="92.0" y="175.0" text-anchor="middle" class="field-label">reg</text>
  <text x="92.0" y="189.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[130:127]</text>
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
  <rect x="728" y="215.0" width="120" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="788.0" y="233.0" text-anchor="middle" class="field-label">add</text>
  <text x="788.0" y="244.0" text-anchor="middle" class="field-label">sub</text>
  <text x="788.0" y="255.0" text-anchor="middle" class="field-label">src</text>
  <text x="788.0" y="266.0" text-anchor="middle" class="field-label">b</text>
  <text x="788.0" y="280.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[68:63]</text>
  <rect x="680" y="215.0" width="48" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="704.0" y="238.5" text-anchor="middle" class="field-label">lr</text>
  <text x="704.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="704.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="704.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[70:69]</text>
  <rect x="632" y="215.0" width="48" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="656.0" y="244.0" text-anchor="middle" class="field-label">post</text>
  <text x="656.0" y="255.0" text-anchor="middle" class="field-label">fn</text>
  <text x="656.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[72:71]</text>
  <rect x="608" y="215.0" width="24" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="620.0" y="238.5" text-anchor="middle" class="field-label">full</text>
  <text x="620.0" y="249.5" text-anchor="middle" class="field-label">xmem</text>
  <text x="620.0" y="260.5" text-anchor="middle" class="field-label">row</text>
  <text x="620.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[73:73]</text>
  <rect x="512" y="215.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="560.0" y="244.0" text-anchor="middle" class="field-label">cr</text>
  <text x="560.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="560.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[77:74]</text>
  <rect x="488" y="215.0" width="24" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="500.0" y="244.0" text-anchor="middle" class="field-label">agg</text>
  <text x="500.0" y="255.0" text-anchor="middle" class="field-label">mode</text>
  <text x="500.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[78:78]</text>
  <rect x="392" y="215.0" width="96" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="440.0" y="244.0" text-anchor="middle" class="field-label">activation</text>
  <text x="440.0" y="255.0" text-anchor="middle" class="field-label">fn</text>
  <text x="440.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[82:79]</text>
  <rect x="320" y="215.0" width="72" height="80" fill="#9B59B6" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="356.0" y="238.5" text-anchor="middle" class="field-label">aaq</text>
  <text x="356.0" y="249.5" text-anchor="middle" class="field-label">inst</text>
  <text x="356.0" y="260.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="356.0" y="274.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[85:83]</text>
  <rect x="272" y="215.0" width="48" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="296.0" y="244.0" text-anchor="middle" class="field-label">vertical</text>
  <text x="296.0" y="255.0" text-anchor="middle" class="field-label">stride</text>
  <text x="296.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[87:86]</text>
  <rect x="176" y="215.0" width="96" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="224.0" y="244.0" text-anchor="middle" class="field-label">lr</text>
  <text x="224.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="224.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[91:88]</text>
  <rect x="104" y="215.0" width="72" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="140.0" y="244.0" text-anchor="middle" class="field-label">horizontal</text>
  <text x="140.0" y="255.0" text-anchor="middle" class="field-label">stride</text>
  <text x="140.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[94:92]</text>
  <rect x="80" y="215.0" width="24" height="80" fill="#45B7D1" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="244.0" text-anchor="middle" class="field-label">aaq</text>
  <text x="92.0" y="255.0" text-anchor="middle" class="field-label">reg</text>
  <text x="92.0" y="269.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[96:95]</text>
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
  <rect x="728" y="295.0" width="120" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="788.0" y="313.0" text-anchor="middle" class="field-label">add</text>
  <text x="788.0" y="324.0" text-anchor="middle" class="field-label">sub</text>
  <text x="788.0" y="335.0" text-anchor="middle" class="field-label">src</text>
  <text x="788.0" y="346.0" text-anchor="middle" class="field-label">b</text>
  <text x="788.0" y="360.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[36:31]</text>
  <rect x="680" y="295.0" width="48" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="704.0" y="318.5" text-anchor="middle" class="field-label">lr</text>
  <text x="704.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="704.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="704.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[38:37]</text>
  <rect x="584" y="295.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="632.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="632.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="632.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[42:39]</text>
  <rect x="488" y="295.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="536.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="536.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="536.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[46:43]</text>
  <rect x="344" y="295.0" width="144" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="416.0" y="313.0" text-anchor="middle" class="field-label">add</text>
  <text x="416.0" y="324.0" text-anchor="middle" class="field-label">sub</text>
  <text x="416.0" y="335.0" text-anchor="middle" class="field-label">src</text>
  <text x="416.0" y="346.0" text-anchor="middle" class="field-label">b</text>
  <text x="416.0" y="360.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[52:47]</text>
  <rect x="296" y="295.0" width="48" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="320.0" y="318.5" text-anchor="middle" class="field-label">lr</text>
  <text x="320.0" y="329.5" text-anchor="middle" class="field-label">inst</text>
  <text x="320.0" y="340.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="320.0" y="354.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[54:53]</text>
  <rect x="200" y="295.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="248.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="248.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="248.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[58:55]</text>
  <rect x="104" y="295.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="152.0" y="324.0" text-anchor="middle" class="field-label">lr</text>
  <text x="152.0" y="335.0" text-anchor="middle" class="field-label">reg</text>
  <text x="152.0" y="349.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[62:59]</text>
  <rect x="80" y="295.0" width="24" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="313.0" text-anchor="middle" class="field-label">add</text>
  <text x="92.0" y="324.0" text-anchor="middle" class="field-label">sub</text>
  <text x="92.0" y="335.0" text-anchor="middle" class="field-label">src</text>
  <text x="92.0" y="346.0" text-anchor="middle" class="field-label">b</text>
  <text x="92.0" y="360.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[68:63]</text>
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
  <rect x="728" y="375.0" width="120" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="788.0" y="404.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="788.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="788.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[4:0]</text>
  <rect x="608" y="375.0" width="120" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="668.0" y="404.0" text-anchor="middle" class="field-label">lcr</text>
  <text x="668.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="668.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[9:5]</text>
  <rect x="368" y="375.0" width="240" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="488.0" y="404.0" text-anchor="middle" class="field-label">label</text>
  <text x="488.0" y="415.0" text-anchor="middle" class="field-label">token</text>
  <text x="488.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[19:10]</text>
  <rect x="296" y="375.0" width="72" height="80" fill="#98D8C8" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="332.0" y="398.5" text-anchor="middle" class="field-label">cond</text>
  <text x="332.0" y="409.5" text-anchor="middle" class="field-label">inst</text>
  <text x="332.0" y="420.5" text-anchor="middle" class="field-label">opcode</text>
  <text x="332.0" y="434.5" text-anchor="middle" class="field-label" style="font-weight: bold;">[22:20]</text>
  <rect x="200" y="375.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="248.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="248.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="248.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[26:23]</text>
  <rect x="104" y="375.0" width="96" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="152.0" y="404.0" text-anchor="middle" class="field-label">lr</text>
  <text x="152.0" y="415.0" text-anchor="middle" class="field-label">reg</text>
  <text x="152.0" y="429.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[30:27]</text>
  <rect x="80" y="375.0" width="24" height="80" fill="#FFA07A" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="92.0" y="393.0" text-anchor="middle" class="field-label">add</text>
  <text x="92.0" y="404.0" text-anchor="middle" class="field-label">sub</text>
  <text x="92.0" y="415.0" text-anchor="middle" class="field-label">src</text>
  <text x="92.0" y="426.0" text-anchor="middle" class="field-label">b</text>
  <text x="92.0" y="440.0" text-anchor="middle" class="field-label" style="font-weight: bold;">[36:31]</text>
  <text x="80" y="475.0" class="inst-label" style="font-weight: bold;">Instruction Type Colors:</text>
  <rect x="80" y="480.0" width="12" height="12" fill="#FFD93D" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="492.0" class="field-label" style="font-size: 9px;">BreakInst (Break / Debug)</text>
  <rect x="80" y="495.0" width="12" height="12" fill="#FF6B6B" stroke="black" stroke-width="1" opacity="0.85"/>
  <text x="100" y="507.0" class="field-label" style="font-size: 9px;">XmemInst (Extended Memory)</text>
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

---


## XMEM Instructions

Memory access instructions for loading and storing data between registers and memory.

### `STR_ACC_REG` — Store Accumulator

**Syntax:** `STR_ACC_REG offset, base`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15) |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR14) |

**General description:**
Store accumulator to memory.

**Pseudo code:**
`Memory[offset + base] = R_ACC`

**Example of usage:**
```asm
STR_ACC_REG CR0, CR1;;
```

### `LDR_MULT_REG` — Load Register

**Syntax:** `LDR_MULT_REG dest, offset, base`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`MultStageReg`](operand-types.md#multstagereg) | `dest`: **`R0`** \| **`R1`** — mult-stage register to load (2-bit field; only these encodings are valid). |
| `offset` | [`LrIdx`](operand-types.md#lridx) | `offset`: **`LR0`**…**`LR15`** — offset register. |
| `base` | [`CrIdx`](operand-types.md#cridx) | `base`: **`CR0`**…**`CR15`** — base address register. |

**General description:**
Load data from memory into a multiplication stage register.

**Pseudo code:**
`dest = Memory[offset + base]  # 128 elements (512 in wide-vector debug mode)`

**Example of usage:**
```asm
SET LR0, CR1;;
LDR_MULT_REG R0, LR0, CR0;;
```

### `LDR_CYCLIC_MULT_REG` — Load Cyclic Register

**Syntax:** `LDR_CYCLIC_MULT_REG offset, base, index`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15) |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR14) |
| `index` | [`LrIdx`](operand-types.md#lridx) | index: Index inside cyclic register (LR0–LR15) |

**General description:**
Load with cyclic addressing into R_CYCLIC.

**Pseudo code:**
`R_CYCLIC[index % 512:128] = Memory[offset + base]`


### `LDR_MULT_MASK_REG` — Load Mask Register

**Syntax:** `LDR_MULT_MASK_REG offset, base, mask_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15) |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR14) |

**General description:**
Load mask data from memory.

**Pseudo code:**
`R_MASK = Memory[offset + base]`


### `XMEM_NOP` — No Operation (XMEM)

**Syntax:** `XMEM_NOP`

**General description:**
No operation for xmem slot.


### `STR_POST_AAQ_REG` — Store Post-AAQ register

**Syntax:** `STR_POST_AAQ_REG offset, base`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: Offset register (LR0–LR15) |
| `base` | [`CrIdx`](operand-types.md#cridx) | base: Base address register (CR0–CR14) |

**General description:**
Write **512 bytes** of **`POST_AAQ_REG`** to external memory. **Interim:** the buffer is **512 bytes** (128×32-bit lanes) until quantization export is finalized.

**Pseudo code:**
`Memory[offset + base] = POST_AAQ_REG (512 bytes); interim staging register`

**Example of usage:**
```asm
STR_POST_AAQ_REG LR0, CR0;;
```

---

## MULT Instructions

Multiplication instructions for element-wise and element-vector operations.
The multiplication result (`mult_result`) is forwarded to the ACC stage in the CPU and not stored in any register in the way.

### `MULT.EE` — Element-wise Multiply

**Syntax:** `MULT.EE ra, cyclic_offset, mask_offset, mask_shift`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `ra` | [`MultStageReg`](operand-types.md#multstagereg) | `ra`: **`R0`** \| **`R1`** — multiplicand mult-stage register (same cycle as `LDR_MULT_REG` into **`R0`**/**`R1`** is allowed). |
| `cyclic_offset` | [`LrIdx`](operand-types.md#lridx) | `cyclic_offset`: **`LR0`**…**`LR15`** — base byte offset into **`R_CYCLIC`**. |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] selecting one of seven masks generated by sequential shift-and-AND with the partition vector. |

**General description:**
Multiply elements of two registers element by element.

**Pseudo code:**
`For each lane i: MULT_RES[i] = ipu_mult(ra[i], R_CYCLIC[cyclic_offset + i]); then apply mask and shift.`

**Example of usage:**
```asm
MULT.EE R0, LR0, 0, LR2;;
```

### `MULT.VE.CYCLIC` — Vector-Element Multiply (cyclic RC)

**Syntax:** `MULT.VE.CYCLIC cyclic_offset, mask_offset, mask_shift, fixed_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `cyclic_offset` | [`LrIdx`](operand-types.md#lridx) | `cyclic_offset`: **`LR0`**…**`LR15`** — base byte offset into **`R_CYCLIC`** (reduced mod 512). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] selecting one of seven masks generated by sequential shift-and-AND with the partition vector. |
| `fixed_idx` | [`LrIdx`](operand-types.md#lridx) | `fixed_idx`: **`LR0`**…**`LR15`** (value read live) — scalar index into **`R0`**/**`R1`**. |

**General description:**
Multiply a fixed element from R0 or R1 against R_CYCLIC[cyclic_offset:cyclic_offset+128]. `fixed_idx` 0..127 selects `R0[fixed_idx]`, 128..255 selects `R1[fixed_idx - 128]`. R_CYCLIC is addressed cyclically modulo 512 elements (no padding with 1 past the boundary).

**Pseudo code:**
`For i in [0, 128): rb = R_CYCLIC[(cyclic_offset + i) % 512]; scalar from R0/R1 via fixed_idx; MULT_RES[i] = scalar * rb (then mask/shift).`

**Example of usage:**
```asm
MULT.VE.CYCLIC LR0, 0, LR2, LR3;;
```

### `MULT.VE.PADDED` — Vector-Element Multiply (padded RC)

**Syntax:** `MULT.VE.PADDED cyclic_offset, mask_offset, mask_shift, fixed_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `cyclic_offset` | [`LrIdx`](operand-types.md#lridx) | `cyclic_offset`: **`LR0`**…**`LR15`** — base byte offset into `R_CYCLIC`; out-of-range lanes use dtype 1. |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in `R_MASK`. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] selecting one of seven masks generated by sequential shift-and-AND with the partition vector. |
| `fixed_idx` | [`LrIdx`](operand-types.md#lridx) | `fixed_idx`: **`LR0`**…**`LR15`** (value read live) — scalar index into **`R0`**/**`R1`**. |

**General description:**
Same scalar × RC row as `MULT.VE.CYCLIC`, but indices at or past the 512-byte RC boundary within the 128-element window use a dtype-specific 1 instead of wrapping.

**Pseudo code:**
`For i in [0, 128): rb = R_CYCLIC[cyclic_offset + i] if in bounds else dtype_one; scalar from R0/R1; MULT_RES[i] = scalar * rb (then mask/shift).`

**Example of usage:**
```asm
MULT.VE.PADDED LR0, 0, LR2, LR3;;
```

### `MULT_NOP` — No Operation (MULT)

**Syntax:** `MULT_NOP`

**General description:**
No operation for multiply slot.


### `MULT.VE.CR` — Vector-Element Multiply (CR scalar)

**Syntax:** `MULT.VE.CR cyclic_offset, mask_offset, mask_shift, cr_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `cyclic_offset` | [`LrIdx`](operand-types.md#lridx) | cyclic_offset: Base offset into RC (cyclic register); non-cyclic — out-of-bounds elements are padded with 1 |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | mask_offset: Immediate mask slot 0–7 (128-bit slice of R_MASK) |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | mask_shift: index ∈ [−3, +3] selecting one of seven masks via sequential shift-and-AND with the partition vector |
| `cr_idx` | [`CrIdx`](operand-types.md#cridx) | cr_idx: CR register whose low byte supplies the fixed scalar multiplier (CR0–CR14) |

**General description:**
Multiply each element of RC[cyclic_offset:cyclic_offset+128] by a scalar from a CR register. Elements beyond RC boundary are treated as 1 (dtype-specific).

**Pseudo code:**
`For i in [0,128): rb = RC[cyclic_offset+i] if in bounds else dtype_one; MULT_RES[i] = CR[cr_idx][0] * rb`

**Example of usage:**
```asm
MULT.VE.CR LR0, 0, LR15, CR3;;
```

### `MULT.VE.AAQ` — Vector-Element Multiply (AAQ scalar)

**Syntax:** `MULT.VE.AAQ cyclic_offset, mask_offset, mask_shift, aaq_rf_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `cyclic_offset` | [`LrIdx`](operand-types.md#lridx) | cyclic_offset: Base offset into RC (cyclic register); non-cyclic — out-of-bounds elements are padded with 1 |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | mask_offset: Immediate mask slot 0–7 (128-bit slice of R_MASK) |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | mask_shift: index ∈ [−3, +3] selecting one of seven masks via sequential shift-and-AND with the partition vector |
| `aaq_rf_idx` | [`AaqRegIdx`](operand-types.md#aaqregidx) | aaq_rf_idx: AAQ register whose low byte supplies the fixed scalar multiplier (AAQ0–AAQ3) |

**General description:**
Multiply each element of RC[cyclic_offset:cyclic_offset+128] by a scalar from an AAQ register. Elements beyond RC boundary are treated as 1 (dtype-specific).

**Pseudo code:**
`For i in [0,128): rb = RC[cyclic_offset+i] if in bounds else dtype_one; MULT_RES[i] = AAQ[aaq_rf_idx][0] * rb`

**Example of usage:**
```asm
MULT.VE.AAQ LR0, 0, LR15, AAQ1;;
```

### `MULT.EE.RR` — Multi-Element Multiply (register by register)

**Syntax:** `MULT.EE.RR ra, mask_offset, mask_shift`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `ra` | [`MultStageReg`](operand-types.md#multstagereg) | `ra`: **`R0`** \| **`R1`** — selects the MEE mode; the chosen register is both multiplicand and multiplier (same cycle as `LDR_MULT_REG` into **`R0`**/**`R1`** is allowed). |
| `mask_offset` | [`MultMaskOffsetImmediate`](operand-types.md#multmaskoffsetimmediate) | `mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`r_mask`**. |
| `mask_shift` | [`LrIdx`](operand-types.md#lridx) | `mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] selecting one of seven masks generated by sequential shift-and-AND with the partition vector. |

**General description:**
Multi-element execution (MEE): multiply a mult-stage register element by element against itself. `ra` selects the execution mode — **`R0`** gives r0-by-r0, **`R1`** gives r1-by-r1.

**Pseudo code:**
`For each lane i: mult_res[i] = ipu_mult(ra[i], ra[i]); then apply mask and shift.`

**Example of usage:**
```asm
MULT.EE.RR R0, 0, LR2;;
```

---

## ACC Instructions

Accumulation instructions for combining values with optional masking and shifting.

### `ACC` — Accumulate

**Syntax:** `ACC`

**General description:**
Accumulate multiply result.

**Pseudo code:**
`R_ACC += multiply_result`


### `ACC.FIRST` — Accumulate First

**Syntax:** `ACC.FIRST`

**General description:**
Set accumulator to multiply result (do not ADD to previous R_ACC).

**Pseudo code:**
`R_ACC = multiply_result`

**Example of usage:**
```asm
ACC.FIRST;;
```

### `RESET_ACC` — Reset Accumulator

**Syntax:** `RESET_ACC`

**General description:**
Reset accumulator to zero.

**Pseudo code:**
`R_ACC = 0`


### `ACC_NOP` — No Operation (ACC)

**Syntax:** `ACC_NOP`

**General description:**
No operation for accumulator slot.


### `ACC.ADD_AAQ` — Accumulate and Add AAQ

**Syntax:** `ACC.ADD_AAQ aaq_rf_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `aaq_rf_idx` | [`AaqRegIdx`](operand-types.md#aaqregidx) | aaq_rf_idx: AAQ register index (AAQ0–AAQ3) |

**General description:**
Accumulate multiply result, then ADD the selected AAQ register (32-bit) to each of the 128 accumulator words.

**Pseudo code:**
`R_ACC += multiply_result;
for i in [0, 128): R_ACC[i] += AAQ_REGS[aaq_rf_idx]`

**Example of usage:**
```asm
ACC.ADD_AAQ AAQ0;;
```

### `ACC.ADD_AAQ.FIRST` — Accumulate and Add AAQ (First)

**Syntax:** `ACC.ADD_AAQ.FIRST aaq_rf_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `aaq_rf_idx` | [`AaqRegIdx`](operand-types.md#aaqregidx) | aaq_rf_idx: AAQ register index (AAQ0–AAQ3) |

**General description:**
Set accumulator to multiply result plus selected AAQ register (do not ADD to previous R_ACC).

**Pseudo code:**
`R_ACC = multiply_result;
for i in [0, 128): R_ACC[i] += AAQ_REGS[aaq_rf_idx]`

**Example of usage:**
```asm
ACC.ADD_AAQ.FIRST AAQ0;;
```

### `ACC.MAX` — Accumulator Max

**Syntax:** `ACC.MAX aaq_rf_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `aaq_rf_idx` | [`AaqRegIdx`](operand-types.md#aaqregidx) | aaq_rf_idx: AAQ register index (AAQ0–AAQ3) |

**General description:**
For each element, SET R_ACC[i] = max(R_ACC[i], MULT_RES[i], AAQ_REGS[aaq_rf_idx]).

**Pseudo code:**
`for i in [0, 128): R_ACC[i] = max(R_ACC[i], MULT_RES[i], AAQ_REGS[aaq_rf_idx])`

**Example of usage:**
```asm
ACC.MAX AAQ0;;
```

### `ACC.MAX.FIRST` — Accumulator Max (First)

**Syntax:** `ACC.MAX.FIRST aaq_rf_idx`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `aaq_rf_idx` | [`AaqRegIdx`](operand-types.md#aaqregidx) | aaq_rf_idx: AAQ register index (AAQ0–AAQ3) |

**General description:**
For each element, SET R_ACC[i] = max(MULT_RES[i], AAQ_REGS[aaq_rf_idx]). Previous R_ACC is ignored (treated as 0).

**Pseudo code:**
`for i in [0, 128): R_ACC[i] = max(MULT_RES[i], AAQ_REGS[aaq_rf_idx])`

**Example of usage:**
```asm
ACC.MAX.FIRST AAQ0;;
```

### `ACC.STRIDE` — Accumulator Stride

**Syntax:** `ACC.STRIDE elements_in_row, horizontal_stride, vertical_stride, offset`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `elements_in_row` | [`ElementsInRow`](operand-types.md#elementsinrow) | elements_in_row: Elements per row (8, 16, 32, or 64) |
| `horizontal_stride` | [`HorizontalStride`](operand-types.md#horizontalstride) | horizontal_stride: Horizontal stride mode (enabled, inverted, expand) |
| `vertical_stride` | [`VerticalStride`](operand-types.md#verticalstride) | vertical_stride: Vertical stride mode (enabled, inverted) |
| `offset` | [`LrIdx`](operand-types.md#lridx) | offset: LR register; value % 4 gives start index in RACC (0, 32, 64, or 96) |

**General description:**
Reorder the multiplication result into R_ACC using horizontal/vertical stride decimation. Only updates the RACC indexes written; leaves the rest unchanged.

**Pseudo code:**
`Decimate MULT_RES as rows×cols; apply horizontal stride (take every 2nd column, optional expand); then vertical stride (take every 2nd row). Write result into R_ACC[start:start+N] where start = (offset%4)*32, N = 32|64|128.`

**Example of usage:**
```asm
ACC.STRIDE 8, off, off, LR0;;
```

---

## AAQ Instructions

Activation and quantization: aggregate r_acc into AAQ registers; ACTIVATE writes activated lanes from r_acc into POST_AAQ_REG; AAQ quantizes POST_AAQ_REG.

### `AAQ_NOP` — No Operation (AAQ)

**Syntax:** `AAQ_NOP`

**General description:**
No operation for AAQ slot.


### `AGG` — Accumulator Aggregate

**Syntax:** `AGG agg_mode, post_fn, cr_idx, aaq_rf_idx, full_xmem_row`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `agg_mode` | [`AggMode`](operand-types.md#aggmode) | agg_mode: sum or max |
| `post_fn` | [`PostFn`](operand-types.md#postfn) | post_fn: value, value_cr, inv, or inv_sqrt |
| `cr_idx` | [`CrIdx`](operand-types.md#cridx) | cr_idx: CR register for value_cr post function (CR0–CR14) |
| `aaq_rf_idx` | [`AaqRegIdx`](operand-types.md#aaqregidx) | aaq_rf_idx: AAQ register to store result (AAQ0–AAQ3) |
| `full_xmem_row` | [`FullXmemRow`](operand-types.md#fullxmemrow) | full_xmem_row: 1 = always 128 lanes; 0 = use CR15.valid_elements (default 0) |

**General description:**
Collapse R_ACC lanes into one value (SUM or MAX); apply post function; store to selected AAQ register. ``full_xmem_row=1`` always uses 128 lanes; ``full_xmem_row=0`` uses CR15.valid_elements.

**Pseudo code:**
`Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128). If sum: v = sum(R_ACC[0..n-1]). If max: v = max(R_ACC[0..n-1], AAQ[aaq_rf_idx]). Apply post_fn(v): value→v, value_cr→v*cr[cr_idx], inv→1/v, inv_sqrt→1/sqrt(v). AAQ[aaq_rf_idx] = result.`

**Example of usage:**
```asm
AGG sum, value, CR0, AAQ0, 0;;
```

### `AGG.FIRST` — Accumulator Aggregate First

**Syntax:** `AGG.FIRST agg_mode, post_fn, cr_idx, aaq_rf_idx, full_xmem_row`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `agg_mode` | [`AggMode`](operand-types.md#aggmode) | agg_mode: sum or max |
| `post_fn` | [`PostFn`](operand-types.md#postfn) | post_fn: value, value_cr, inv, or inv_sqrt |
| `cr_idx` | [`CrIdx`](operand-types.md#cridx) | cr_idx: CR register for value_cr post function (CR0–CR14) |
| `aaq_rf_idx` | [`AaqRegIdx`](operand-types.md#aaqregidx) | aaq_rf_idx: AAQ register to store result (AAQ0–AAQ3) |
| `full_xmem_row` | [`FullXmemRow`](operand-types.md#fullxmemrow) | full_xmem_row: 1 = always 128 lanes; 0 = use CR15.valid_elements (default 0) |

**General description:**
Like AGG, but for MAX mode ignores the previous AAQ register value, avoiding contamination from uninitialized data. ``full_xmem_row=1`` always uses 128 lanes; ``full_xmem_row=0`` uses CR15.valid_elements.

**Pseudo code:**
`Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128). If sum: v = sum(R_ACC[0..n-1]). If max: v = max(R_ACC[0..n-1]) (previous AAQ value is NOT included). Apply post_fn(v): value→v, value_cr→v*cr[cr_idx], inv→1/v, inv_sqrt→1/sqrt(v). AAQ[aaq_rf_idx] = result.`

**Example of usage:**
```asm
AGG.FIRST max, value, CR0, AAQ0, 0;;
```

### `AAQ` — AAQ Quantize

**Syntax:** `AAQ full_xmem_row`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `full_xmem_row` | [`FullXmemRow`](operand-types.md#fullxmemrow) | full_xmem_row: 1 = always 128 lanes (full XMEM row); 0 = use CR15.valid_elements lane count |

**General description:**
Quantize wide lanes in **`POST_AAQ_REG`** (INT32 per lane in INT8 mode) to INT8, storing clamped bytes in the **leading 128 bytes** of **`POST_AAQ_REG`** and clearing the rest of the register. Wide lanes are normally produced by **`ACTIVATE`** (from ``r_acc``). Requires INT8 mode. ``full_xmem_row=1`` always processes all 128 lanes; ``full_xmem_row=0`` uses ``CR15.valid_elements`` as the active lane count.

**Pseudo code:**
`Requires INT8 mode (IpuState.dtype == DType.INT8 in the Python emulator). Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128). For i in [0, n): POST_AAQ_REG[i] = clamp(trunc(POST_AAQ_REG wide lane i), -128, 127). POST_AAQ_REG[n..511] = 0.`

**Example of usage:**
```asm
AAQ 1;;
```

### `ACTIVATE` — Accumulator Activation

**Syntax:** `ACTIVATE activation_fn, full_xmem_row`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `activation_fn` | [`ActivationFn`](operand-types.md#activationfn) | activation_fn: keyword naming the activation (one of identity, relu, relu6, sigmoid, tanh, gelu, softplus, elu, exp2; see ACTIVATION_FN_NAMES) |
| `full_xmem_row` | [`FullXmemRow`](operand-types.md#fullxmemrow) | full_xmem_row: 1 = always 128 lanes; 0 = use CR15.valid_elements (default 0) |

**General description:**
Read active lanes from ``r_acc``, apply the selected element-wise activation, and write results into the same lane indices of ``POST_AAQ_REG`` (``r_acc`` is unchanged). ``full_xmem_row=1`` always activates all 128 lanes; ``full_xmem_row=0`` uses CR15.valid_elements. The activation is selected by keyword (see ACTIVATION_FN_NAMES). The available activation functions are: ``identity`` (0), ``relu`` (1), ``relu6`` (2), ``sigmoid`` (3), ``tanh`` (4), ``gelu`` (5), ``softplus`` (6), ``elu`` (7), ``exp2`` (8), ``reciprocal`` (9), ``rsqrt`` (10). For Python emulator calibration (virtual α), see docs/content/building-applications.md#activations-emulator.

**Pseudo code:**
`Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128) and k = encoded activation index. For i in [0, n): POST_AAQ_REG[i] = activation_k(R_ACC[i]) (same 32-bit lane format as R_ACC). R_ACC is not modified. The selector uses four bits; encodings outside the eleven named activations behave as identity. α for elu is not an ISA operand; see docs/content/building-applications.md#activations-emulator.`

**Example of usage:**
```asm
ACTIVATE relu, 0;;
```

---

## LR Instructions

Loop register manipulation instructions for controlling loop counters and addresses.

### `SET` — Set Loop Register

**Syntax:** `SET reg, src`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LrIdx`](operand-types.md#lridx) | reg: Loop register (LR0–LR15) |
| `src` | [`CrIdx`](operand-types.md#cridx) | src: Source configuration register (CR0–CR14) |

**General description:**
Copy a 32-bit value from a configuration register into a loop register.

**Pseudo code:**
`reg = cr[src]`

**Example of usage:**
```asm
SET LR0, CR1;;
```

### `ADD` — Add

**Syntax:** `ADD dest, src_a, src_b`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination local register (LR0–LR15) |
| `src_a` | [`LrIdx`](operand-types.md#lridx) | src_a: First source local register (LR0–LR15) |
| `src_b` | [`AddSubSrcB`](operand-types.md#addsubsrcb) | src_b: Second source — LR0–LR15, CR0–CR14, or unsigned immediate 0–31 |

**General description:**
Add two sources (second source may be an LR, CR, or 5-bit unsigned immediate) and store the result in the destination LR.

**Pseudo code:**
`dest = src_a + src_b`

**Example of usage:**
```asm
ADD LR0, LR1, LR2;;
ADD LR3, LR1, CR5;;
ADD LR4, LR1, 7;;
```

### `SUB` — Subtract

**Syntax:** `SUB dest, src_a, src_b`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dest: Destination local register (LR0–LR15) |
| `src_a` | [`LrIdx`](operand-types.md#lridx) | src_a: First source local register (LR0–LR15) |
| `src_b` | [`AddSubSrcB`](operand-types.md#addsubsrcb) | src_b: Second source — LR0–LR15, CR0–CR14, or unsigned immediate 0–31 |

**General description:**
Subtract the second source from the first (second source may be an LR, CR, or 5-bit unsigned immediate) and store the result in the destination LR.

**Pseudo code:**
`dest = src_a - src_b`

**Example of usage:**
```asm
SUB LR0, LR1, LR2;;
SUB LR3, LR1, CR5;;
SUB LR4, LR1, 7;;
```

### `INCR_MOD_POW2` — Increment Loop Register Modulo Power of Two

**Syntax:** `INCR_MOD_POW2 dst, step, k`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `dest` | [`LrIdx`](operand-types.md#lridx) | dst: Destination loop register (LR0–LR15); read and written |
| `step` | [`LcrIdx`](operand-types.md#lcridx) | step: Signed 32-bit increment from LR0–LR15 or CR0–CR14 |
| `k` | [`LrModPow2KImmediate`](operand-types.md#lrmodpow2kimmediate) | k: Immediate in [1, 9]; encoded in 4 bits as (k − 1); mask = (1 << k) - 1 |

**General description:**
Add a loop or configuration register into the destination loop register, then mask to k low bits (mod 2^k).

**Pseudo code:**
`dst <- (dst + step) & ((1 << k) - 1)`

**Example of usage:**
```asm
INCR_MOD_POW2 LR2, LR3, 4;;
```

---

## Conditional Branch Instructions

Control flow instructions for branching based on conditions or unconditionally.

### `BEQ` — Branch if Equal

**Syntax:** `BEQ reg1, reg2, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR14) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if two registers are equal.

**Pseudo code:**
`if (reg1 == reg2) PC = label`

**Example of usage:**
```asm
BEQ LR0, LR1, end;;
```

### `BNE` — Branch if Not Equal

**Syntax:** `BNE reg1, reg2, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR14) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if two registers are not equal.

**Pseudo code:**
`if (reg1 != reg2) PC = label`

**Example of usage:**
```asm
BNE LR0, CR0, loop;;
```

### `BLT` — Branch if Less Than

**Syntax:** `BLT reg1, reg2, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg1` | [`LcrIdx`](operand-types.md#lcridx) | reg1: First register to compare (LR0–LR15 or CR0–CR14) |
| `reg2` | [`LcrIdx`](operand-types.md#lcridx) | reg2: Second register to compare (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if first register is less than second.

**Pseudo code:**
`if (reg1 < reg2) PC = label`

**Example of usage:**
```asm
BLT LR0, CR1, smaller;;
```

### `BNZ` — Branch if Not Zero

**Syntax:** `BNZ test_reg, base_reg, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `test_reg` | [`LcrIdx`](operand-types.md#lcridx) | test_reg: Register to test (LR0–LR15 or CR0–CR14) |
| `base_reg` | [`LcrIdx`](operand-types.md#lcridx) | base_reg: Base comparison register (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if test register not equal to base register.

**Pseudo code:**
`if (test_reg != base_reg) PC = label`

**Example of usage:**
```asm
BNZ LR3, LR0, loop;;
```

### `BZ` — Branch if Zero

**Syntax:** `BZ test_reg, base_reg, label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `test_reg` | [`LcrIdx`](operand-types.md#lcridx) | test_reg: Register to test (LR0–LR15 or CR0–CR14) |
| `base_reg` | [`LcrIdx`](operand-types.md#lcridx) | base_reg: Base comparison register (LR0–LR15 or CR0–CR14) |
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Branch if test register equals base register.

**Pseudo code:**
`if (test_reg == base_reg) PC = label`

**Example of usage:**
```asm
BZ LR0, LR1, zero;;
```

### `B` — Unconditional Branch

**Syntax:** `B label`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `label` | [`Label`](operand-types.md#label) | label: Branch target label |

**General description:**
Always branch to label.

**Pseudo code:**
`PC = label`

**Example of usage:**
```asm
B start;;
```

### `BR` — Branch Register

**Syntax:** `BR reg`

**Operands:** *(the **Type** column links to the [operand type reference](operand-types.md))*

| Name | Type | Details |
|------|------|---------|
| `reg` | [`LcrIdx`](operand-types.md#lcridx) | reg: Register containing target address (LR0–LR15 or CR0–CR14) |

**General description:**
Branch to address in register.

**Pseudo code:**
`PC = reg`


### `BKPT` — Breakpoint

**Syntax:** `BKPT`

**General description:**
Conditional breakpoint.

**Pseudo code:**
`Halt execution (debugging)`

---

## Break Instructions

Debug break instructions for halting execution and entering debug mode.

### `BREAK` — Break

**Syntax:** `BREAK`

**General description:**
Unconditional break.

**Pseudo code:**
`Halt execution`


### `BREAK.IFEQ` — Break if Equal

**Syntax:** `BREAK.IFEQ reg, value`

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
BREAK.IFEQ LR0, 10;;
```

### `BREAK_NOP` — No Operation (BREAK)

**Syntax:** `BREAK_NOP`

**General description:**
No operation for BREAK slot.

---
