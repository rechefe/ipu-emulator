"""Custom rule for compiling IPU assembly files."""

def assemble_asm(name, src, output_format = "bin", defines = {}, includes = [], target = "emulator", **kwargs):
    """
    Macro to assemble IPU assembly files.

    Args:
        name: Name of the rule
        src: Source .asm file
        output_format: Output format (default: "bin")
        defines: Template values, {name: value}, for the parameters the
            source's ipu:params header declares; each is passed as
            --define 'name=value' (no single quotes in a value), in sorted
            order so the command is stable. A tuple is written 1,2,3.
        includes: Files the source's {% include %} / {% import %} read; each
            distinct directory is passed as -I, after the source's own.
            They must be source files named by relative path (not labels),
            as the directory is found by resolving each one's symlink.
        target: "emulator" (every slot) or "hardware" (the hardware slots
            only; an emulator-only instruction fails the build).
        **kwargs: Additional arguments passed to genrule
    """
    if target not in ("emulator", "hardware"):
        fail("assemble_asm target must be emulator or hardware, not " + target)
    output_file = name + "." + output_format
    flags = [] if target == "emulator" else ["--target " + target]
    flags += ["--define '%s=%s'" % (key, defines[key]) for key in sorted(defines)]
    include_dirs = []
    for include in includes:
        directory = include.rpartition("/")[0] or "."
        if directory not in include_dirs:
            include_dirs.append(directory)
    # The directory the include really lives in: Bazel stages each file as a
    # symlink, and the assembler rejects an include that resolves outside its root.
    flags += ["-I $$(dirname $$(readlink -f $(location %s)))" % _first_in(includes, d) for d in include_dirs]

    native.genrule(
        name = name,
        srcs = [src] + includes,
        outs = [output_file],
        cmd = " ".join([
            "$(location //src/ipu-as:ipu-as) assemble --input $(location %s)" % src,
            "--output $@ --format " + output_format,
        ] + flags),
        tools = ["//src/ipu-as:ipu-as"],
        **kwargs
    )

def _first_in(includes, directory):
    """The first of `includes` in `directory`, to name that directory with $(location)."""
    for include in includes:
        if (include.rpartition("/")[0] or ".") == directory:
            return include
    fail("no include in " + directory)
