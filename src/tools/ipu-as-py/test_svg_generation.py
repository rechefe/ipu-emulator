#!/usr/bin/env python3
"""
Test script to demonstrate the SVG generation function for instruction fields.
"""

import sys
sys.path.insert(0, '/mnt/c/Users/eyalr/dev/ipu-c-samples/src/tools/ipu-as-py/src')

from ipu_as.compound_inst import CompoundInst

# Generate the SVG
svg_content = CompoundInst.generate_fields_svg()

# Save to file
output_file = "instruction_fields_layout.svg"
with open(output_file, 'w') as f:
    f.write(svg_content)

print(f"SVG generated successfully: {output_file}")
print(f"SVG content preview (first 500 chars):\n{svg_content[:500]}...")
