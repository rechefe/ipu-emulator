"""Cycle Inspector traces as build outputs (E08-16, Q55).

A trace the docs page replays is never checked in: `kernel_trace` records a
registered kernel case with the E04-29 trace writer
(//src/tools/ipu-apps:trace_case), then converts it to the page's JSON
(`ipu-gen inspector-trace`). The docs build regenerates every trace, so the
page cannot show a stale ISA or a stale kernel.
"""

def _shell_word(text):
    """`text` as one single-quoted shell word, with `$` escaped for genrule."""
    return "'" + text.replace("'", "'\\''").replace("$", "$$") + "'"

def kernel_trace(name, kernel, case = "default", options = {}, max_cycles = 5000, visibility = None):
    """`<name>.json`: the Cycle Inspector document of one kernel case.

    Args:
        name: The target, and the document's file name (the page's kernel id).
        kernel: The registered kernel (`bazel run //src/tools/ipu-apps:query`).
        case: One of its cases. A kernel's element formats are its cases
            (fully_connected: `int8`, `fp8_e4m3`, `fp8_e5m2`).
        options: Case option overrides, NAME -> value (JSON-encoded on the
            command line).
        max_cycles: Cycles the document keeps; it says when it is truncated.
        visibility: The target's visibility.
    """
    flags = " ".join([
        "--option " + _shell_word("%s=%s" % (option, json.encode(value)))
        for option, value in sorted(options.items())
    ])
    native.genrule(
        name = name,
        outs = [name + ".json"],
        cmd = " && ".join([
            "TMP=$$(mktemp -d)",
            "trap 'rm -rf \"$$TMP\"' EXIT",
            "$(location //src/tools/ipu-apps:trace_case) %s --case %s %s --trace $$TMP/trace.jsonl --context $$TMP/context.json" % (kernel, case, flags),
            "$(location //src/tools/ipu-gen:ipu-gen) inspector-trace $$TMP/trace.jsonl $$TMP/context.json $@ --max-cycles %d" % max_cycles,
        ]),
        tools = [
            "//src/tools/ipu-apps:trace_case",
            "//src/tools/ipu-gen:ipu-gen",
        ],
        # Built with the docs, not in the default CI tier (E01).
        tags = ["docs"],
        visibility = visibility,
    )
