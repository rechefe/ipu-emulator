"""Automatically load Python dependencies from pyproject.toml"""

def _parse_pyproject_deps_impl(repository_ctx):
    """Parse pyproject.toml and extract dependency names"""
    pyproject = repository_ctx.read(repository_ctx.attr.pyproject_toml)
    
    # Simple parser for [project] dependencies array and docs optional dependencies
    deps = []
    in_deps = False
    in_docs_deps = False
    bracket_count = 0
    
    for line in pyproject.split("\n"):
        line = line.strip()
        if line == "dependencies = [":
            in_deps = True
            continue
        if line == "docs = [":
            in_docs_deps = True
            bracket_count = 1
            continue
        if in_docs_deps and "[" in line and not line.startswith("#"):
            bracket_count += line.count("[")
        if in_docs_deps and "]" in line:
            bracket_count -= line.count("]")
            if bracket_count == 0:
                in_docs_deps = False
                continue
        if in_deps or in_docs_deps:
            if in_deps and line == "]":
                in_deps = False
                continue
            # Extract package name from "package>=version" format
            if line.startswith('"') or line.startswith("'"):
                dep = line.strip('",\'')
                # Get just the package name before any version specifier
                pkg_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
                if pkg_name:
                    deps.append(pkg_name)
    
    # Generate a .bzl file with the list
    content = """# Auto-generated from pyproject.toml
\"\"\"Python dependencies extracted from pyproject.toml\"\"\"

def get_requirements(requirement_fn):
    \"\"\"Returns list of requirement() calls for all dependencies in pyproject.toml\"\"\"
    return [
        requirement_fn("{}"),
    ]
""".format('"),\n        requirement_fn("'.join(deps))
    
    repository_ctx.file("deps.bzl", content)
    repository_ctx.file("BUILD.bazel", "exports_files(['deps.bzl'])")

parse_pyproject_deps = repository_rule(
    implementation = _parse_pyproject_deps_impl,
    attrs = {
        "pyproject_toml": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
    local = True,
)
