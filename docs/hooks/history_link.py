"""MkDocs hook: GitHub commit-history URL for the current page (see issue #59)."""


def on_page_context(context, page, config, **_kwargs):
    repo = (config.get("repo_url") or "").rstrip("/")
    if not repo:
        return context

    extra = config.get("extra") or {}
    branch = extra.get("docs_git_branch", "master")
    rel = page.file.src_path.replace("\\", "/")
    context["history_url"] = f"{repo}/commits/{branch}/docs/{rel}"
    return context
