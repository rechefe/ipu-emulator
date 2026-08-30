"""Custom rule for compiling IPU assembly files."""

def assemble_asm(name, src, output_format = "bin", **kwargs):
    """
    Macro to assemble IPU assembly files.

    Args:
        name: Name of the rule
        src: Source .asm file
        output_format: Output format (default: "bin")
        **kwargs: Additional arguments passed to genrule
    """
    output_file = name + "." + output_format

    native.genrule(
        name = name,
        srcs = [src],
        outs = [output_file],
        cmd = "$(location //src/tools/ipu-as-py:ipu-as) assemble --input $< --output $@ --format " + output_format,
        tools = ["//src/tools/ipu-as-py:ipu-as"],
        **kwargs
    )
