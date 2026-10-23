"""Cycle Inspector traces as build outputs (E08-16, Q55).

A trace the docs page replays is never checked in: `kernel_trace` records a
kernel case, then converts it to the page's JSON (`ipu-gen inspector-trace`).
The docs build regenerates every trace, so the page cannot show a stale ISA or
a stale kernel.

The kernel is one of the kernel tree (`//kernels/<family>/<name>`, or a
registry entry `//kernels/<family>:<name>`), recorded through the harness
runner (`//src/ipu-harness:run_kernel --trace --context`, E07-26).
"""

_RUN_KERNEL = "//src/ipu-harness:run_kernel"
_KERNELS = "//kernels/"

def _shell_word(text):
    """`text` as one single-quoted shell word, with `$` escaped for genrule."""
    return "'" + text.replace("'", "'\\''").replace("$", "$$") + "'"

def _kernel_path(kernel):
    """The run_kernel argument of a //kernels label: the folder, or REGISTRY.toml::ENTRY."""
    package, _, entry = kernel[2:].partition(":")
    if entry and entry != package.rpartition("/")[2]:
        return package + "/registry.toml::" + entry
    return package

def _record(kernel, case, format):
    """The command that writes $TMP/trace.jsonl and $TMP/context.json."""
    if not kernel.startswith(_KERNELS):
        fail("kernel_trace: %s is no //kernels label" % kernel)
    files = "--trace $$TMP/trace.jsonl --context $$TMP/context.json"
    flags = "" if format == None else " --format " + _shell_word(format)
    return "$(location %s) %s --case %s%s %s" % (_RUN_KERNEL, _kernel_path(kernel), _shell_word(case), flags, files)

def kernel_trace(name, kernel, case = "default", format = None, max_cycles = 5000, visibility = None):
    """`<name>.json`: the Cycle Inspector document of one kernel case.

    Args:
        name: The target, and the document's file name (the page's kernel id).
        kernel: The kernel: a //kernels label (`//kernels/conformance/vector_tour`,
            or `//kernels/<family>:<entry>`).
        case: One of its cases, run in its first format unless `format` names
            another.
        format: The element format to record the case in (`run_kernel
            --format`), such as `e4`.
        max_cycles: Cycles the document keeps; it says when it is truncated.
        visibility: The target's visibility.
    """
    record = _record(kernel, case, format)
    native.genrule(
        name = name,
        # The kernel's files (and its family's, or its registry's): the tree.
        srcs = ["//kernels:tree"],
        outs = [name + ".json"],
        cmd = " && ".join([
            "TMP=$$(mktemp -d)",
            "trap 'rm -rf \"$$TMP\"' EXIT",
            record,
            "$(location //src/ipu-gen:ipu-gen) inspector-trace $$TMP/trace.jsonl $$TMP/context.json $@ --max-cycles %d" % max_cycles,
        ]),
        tools = [
            _RUN_KERNEL,
            "//src/ipu-gen:ipu-gen",
        ],
        # Built with the docs, not in the default CI tier (E01).
        tags = ["docs"],
        visibility = visibility,
    )
