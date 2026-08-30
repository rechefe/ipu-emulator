"""Auto-generate requirements.txt from pyproject.toml"""

def _generate_requirements_impl(repository_ctx):
    """Generate requirements.txt from one or more pyproject.toml files using uv"""
    pyprojects = [repository_ctx.path(label) for label in repository_ctx.attr.pyproject_tomls]

    # Run uv pip compile to generate requirements including optional dependencies
    result = repository_ctx.execute(
        [
            "uv",
            "pip",
            "compile",
        ] + [str(pyproject) for pyproject in pyprojects] + [
            "--extra=docs",
            "--extra=dev",
            "--python-version=3.10",
            "--generate-hashes",
        ],
        quiet = False,
    )
    
    if result.return_code != 0:
        fail("Failed to compile requirements from pyproject.toml: " + result.stderr)

    # Drop local editable paths emitted from tool.uv.sources; Bazel wires those deps directly.
    filtered_lines = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("-e ") or stripped.startswith("--editable"):
            continue
        if "../ipu-" in stripped or stripped.startswith("../") and "ipu-" in stripped:
            continue
        filtered_lines.append(line)

    # Write the requirements.txt
    repository_ctx.file("requirements.txt", "\n".join(filtered_lines) + "\n")
    
    # Create a minimal BUILD file
    repository_ctx.file("BUILD.bazel", """
exports_files(["requirements.txt"])
""")

generate_requirements = repository_rule(
    implementation = _generate_requirements_impl,
    attrs = {
        "pyproject_tomls": attr.label_list(
            mandatory = True,
            allow_files = True,
            doc = "The pyproject.toml files to compile",
        ),
    },
    local = True,
    doc = "Generate requirements.txt from pyproject.toml at build time",
)
