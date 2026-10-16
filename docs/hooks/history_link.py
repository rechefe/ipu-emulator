"""MkDocs hook: GitHub commit-history URL for the current page (see issue #59).

A hand-written page links to the history of its source under docs/content/.
A page generated at build time links to the history of its generator, looked
up in `extra.generated_pages` in mkdocs.yml (page path -> generator path).
That map is the one place to update when a generator or a generated page
moves. Any other page outside docs/content/ gets no history link.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.history_link")


def _docs_subdir(config) -> str:
    """The docs_dir's path under docs/, e.g. "content"."""
    docs_path = Path(config.get("docs_dir") or "docs")
    if docs_path.is_absolute():
        return docs_path.name
    return str(docs_path).replace("\\", "/").strip("/")


def _source_in_repo(abs_src: str, docs_subdir: str) -> tuple[Path, str] | None:
    """(repo root, path under docs/<docs_subdir>/) for a page whose real file is in the repo.

    Bazel symlinks hand-written pages to their repo files and generated pages
    to bazel-out, so the resolved path tells the two apart. Returns None for
    a page whose real file is not under docs/<docs_subdir>/.
    """
    posix = Path(abs_src).resolve().as_posix()
    marker = f"/docs/{docs_subdir}/"
    if marker not in posix:
        return None
    root, rel = posix.rsplit(marker, 1)
    return Path(root or "/"), rel


def _generated_pages(config) -> dict[str, str]:
    extra = config.get("extra") or {}
    return dict(extra.get("generated_pages") or {})


def history_path(src_uri: str, abs_src: str, config) -> str | None:
    """Repo path whose history the page links to, or None for no link."""
    generated = _generated_pages(config)
    if src_uri in generated:
        return generated[src_uri]
    docs_subdir = _docs_subdir(config)
    source = _source_in_repo(abs_src, docs_subdir)
    if source is None:
        return None
    return "/".join(("docs", docs_subdir, source[1]))


def on_files(files, config, **_kwargs):
    """Warn when the generated-pages map disagrees with the files on disk."""
    generated = _generated_pages(config)
    docs_subdir = _docs_subdir(config)
    repo_root: Path | None = None
    for file in files.documentation_pages():
        source = _source_in_repo(file.abs_src_path, docs_subdir)
        if source is None:
            continue
        repo_root = source[0]
        if file.src_uri in generated:
            log.warning(
                "history_link: %s is listed in extra.generated_pages but is a file in "
                "docs/%s/; remove it from the map",
                file.src_uri,
                docs_subdir,
            )
    if repo_root is not None:
        for page, generator in generated.items():
            if not (repo_root / generator).is_file():
                log.warning(
                    "history_link: generator %s for %s (extra.generated_pages) does not exist",
                    generator,
                    page,
                )
    return files


def on_page_context(context, page, config, **_kwargs):
    repo = (config.get("repo_url") or "").rstrip("/")
    if not repo:
        return context
    branch = (config.get("extra") or {}).get("docs_git_branch", "master")
    path = history_path(page.file.src_uri, page.file.abs_src_path, config)
    if path is not None:
        context["history_url"] = f"{repo}/commits/{branch}/{path}"
    return context
