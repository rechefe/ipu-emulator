"""Cycle Inspector traces as build outputs (E08-16, Q55).

A trace the docs page replays is never checked in: `kernel_trace` records a
kernel case, then converts it to the page's JSON (`ipu-gen inspector-trace`).
The docs build regenerates every trace, so the page cannot show a stale ISA or
a stale kernel.

A kernel of the kernel tree (`//kernels/<family>/<name>`, or a registry entry
`//kernels/<family>:<name>`) is recorded through the harness runner
(`//src/tools/ipu-harness:run_kernel --trace --context`, E07-26). Any other
name is an `ipu_apps` kernel, recorded with `//src/tools/ipu-apps:trace_case`
until its family migrates (E07); E07-25 deletes that path.
"""

_RUN_KERNEL = "//src/tools/ipu-harness:run_kernel"
_TRACE_CASE = "//src/tools/ipu-apps:trace_case"
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

def _record(kernel, case, options):
    """The command that writes $TMP/trace.jsonl and $TMP/context.json, and its tool."""
    files = "--trace $$TMP/trace.jsonl --context $$TMP/context.json"
    if kernel.startswith(_KERNELS):
        if options:
            fail("kernel_trace: %s runs its cases as kernel.toml declares them; give it a case, not options" % kernel)
        return "$(location %s) %s --case %s %s" % (_RUN_KERNEL, _kernel_path(kernel), case, files), _RUN_KERNEL
    flags = " ".join([
        "--option " + _shell_word("%s=%s" % (option, json.encode(value)))
        for option, value in sorted(options.items())
    ])
    return "$(location %s) %s --case %s %s %s" % (_TRACE_CASE, kernel, case, flags, files), _TRACE_CASE

def kernel_trace(name, kernel, case = "default", options = {}, max_cycles = 5000, visibility = None):
    """`<name>.json`: the Cycle Inspector document of one kernel case.

    Args:
        name: The target, and the document's file name (the page's kernel id).
        kernel: The kernel: a //kernels label (`//kernels/conformance/vector_tour`,
            or `//kernels/<family>:<entry>`), or a registered ipu_apps kernel
            (`bazel run //src/tools/ipu-apps:query`).
        case: One of its cases. An ipu_apps kernel's element formats are its
            cases (fully_connected: `int8`, `fp8_e4m3`, `fp8_e5m2`); a
            //kernels case runs in its first format.
        options: Case option overrides of an ipu_apps kernel, NAME -> value
            (JSON-encoded on the command line).
        max_cycles: Cycles the document keeps; it says when it is truncated.
        visibility: The target's visibility.
    """
    record, recorder = _record(kernel, case, options)
    native.genrule(
        name = name,
        # A //kernels kernel's files (and its family's, or its registry's): the tree.
        srcs = ["//kernels:tree"] if kernel.startswith(_KERNELS) else [],
        outs = [name + ".json"],
        cmd = " && ".join([
            "TMP=$$(mktemp -d)",
            "trap 'rm -rf \"$$TMP\"' EXIT",
            record,
            "$(location //src/tools/ipu-gen:ipu-gen) inspector-trace $$TMP/trace.jsonl $$TMP/context.json $@ --max-cycles %d" % max_cycles,
        ]),
        tools = [
            recorder,
            "//src/tools/ipu-gen:ipu-gen",
        ],
        # Built with the docs, not in the default CI tier (E01).
        tags = ["docs"],
        visibility = visibility,
    )
