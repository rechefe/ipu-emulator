#!/usr/bin/env python3
"""Generate markdown documentation for IPU assembly instructions."""

import sys
from pathlib import Path


def generate_instruction_docs(output_path: Path) -> None:
    """Generate instruction reference documentation."""
    # Import here to avoid circular dependencies
    from ipu_as.inst import Inst
    
    content = ["# IPU Assembly Instruction Reference\n"]
    content.append("This document describes all available IPU assembly instructions.\n")
    
    for inst_class in Inst.get_all_instruction_classes():
        content.append(inst_class.description())
        content.append("\n---\n")
    
    output_path.write_text("\n".join(content))
    print(f"Generated documentation at {output_path}")


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_instruction_docs(output_dir / "instructions.md")
