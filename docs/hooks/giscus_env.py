"""MkDocs hook: merge giscus settings from environment (CI / local).

Set in GitHub Actions (repository secrets or variables), or export before
`mkdocs build` / `bazel build //docs:build_docs`. See docs/mkdocs.yml extra.giscus.
"""

from __future__ import annotations

import os
from typing import Any

_ENV_TO_YAML = (
    ("repo_id", "GISCUS_REPO_ID"),
    ("category_id", "GISCUS_CATEGORY_ID"),
    ("repo", "GISCUS_REPO"),
    ("category", "GISCUS_CATEGORY"),
    ("mapping", "GISCUS_MAPPING"),
    ("reactions_enabled", "GISCUS_REACTIONS_ENABLED"),
    ("emit_metadata", "GISCUS_EMIT_METADATA"),
    ("input_position", "GISCUS_INPUT_POSITION"),
    ("theme", "GISCUS_THEME"),
    ("lang", "GISCUS_LANG"),
    ("strict", "GISCUS_STRICT"),
)


def on_config(config: Any, **_kwargs) -> Any:
    extra = getattr(config, "extra", None)
    if extra is None:
        config.extra = {}
        extra = config.extra
    giscus = dict(extra.get("giscus") or {})

    for yaml_key, env_name in _ENV_TO_YAML:
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        val = str(raw).strip()
        if val:
            giscus[yaml_key] = val

    extra["giscus"] = giscus
    return config
