"""MkDocs hook: GitHub commit-history URL for the current page (see issue #59)."""

from pathlib import Path


def _path_under_docs_package(abs_src: Path, docs_subdir: str) -> str:
    """Return path under <docs_subdir>/ for GitHub URLs (repo or Bazel temp tree)."""
    posix = abs_src.resolve().as_posix()
    docs_subdir = docs_subdir.replace("\\", "/").strip("/")
    suffix = f"/{docs_subdir}/"
    if suffix in posix:
        return posix.rsplit(suffix, 1)[1]

    raw = Path(abs_src).as_posix().replace("\\", "/")
    if not raw.startswith("/"):
        return raw

    return Path(raw).name


def on_page_context(context, page, config, **_kwargs):
    repo = (config.get("repo_url") or "").rstrip("/")
    if not repo:
        return context

    extra = config.get("extra") or {}
    branch = extra.get("docs_git_branch", "master")
    raw_docs_dir = config.get("docs_dir") or "docs"
    docs_path = Path(raw_docs_dir)
    if docs_path.is_absolute():
        docs_subdir = docs_path.name
    else:
        docs_subdir = str(docs_path).replace("\\", "/").strip("/")

    rel = _path_under_docs_package(Path(page.file.abs_src_path), docs_subdir)
    repo_path = "/".join(("docs", docs_subdir, rel))
    context["history_url"] = f"{repo}/commits/{branch}/{repo_path}"
    return context
