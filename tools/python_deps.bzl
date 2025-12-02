"""Auto-generate requirements.txt from pyproject.toml"""

def _generate_requirements_impl(repository_ctx):
    """Generate requirements.txt from pyproject.toml using uv"""
    pyproject = repository_ctx.path(repository_ctx.attr.pyproject_toml)
    
    # Run uv pip compile to generate requirements
    result = repository_ctx.execute(
        ["uv", "pip", "compile", str(pyproject), "--generate-hashes"],
        quiet = False,
    )
    
    if result.return_code != 0:
        fail("Failed to compile requirements from pyproject.toml: " + result.stderr)
    
    # Write the requirements.txt
    repository_ctx.file("requirements.txt", result.stdout)
    
    # Create a minimal BUILD file
    repository_ctx.file("BUILD.bazel", """
exports_files(["requirements.txt"])
""")

generate_requirements = repository_rule(
    implementation = _generate_requirements_impl,
    attrs = {
        "pyproject_toml": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "The pyproject.toml file to compile",
        ),
    },
    local = True,
    doc = "Generate requirements.txt from pyproject.toml at build time",
)
