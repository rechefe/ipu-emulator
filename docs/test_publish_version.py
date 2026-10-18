"""The docs version publisher against a temporary git repository (E09-21)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from mike.versions import Versions

sys.path.insert(0, str(Path(__file__).parent))

import publish_version  # noqa: E402

PAGES = {"index.html": "<h1>home</h1>", "reference/isa/instructions/index.html": "<h1>isa</h1>"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def _site(tmp: Path, name: str, pages: dict[str, str]) -> Path:
    path = tmp / f"{name}.tar"
    with tarfile.open(path, "w") as tar:
        for page, text in pages.items():
            data = text.encode()
            info = tarfile.TarInfo(f"./{page}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clone with an origin (a bare repository) that has no gh-pages yet."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    for key, value in (("user.name", "test"), ("user.email", "test@example.com")):
        _git(work, "config", key, value)
    (work / "README").write_text("x\n")
    _git(work, "add", "README")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "push", "-q", "origin", "HEAD:master")
    return work


def _branch_file(repo: Path, path: str) -> str:
    return _git(repo, "show", f"origin/gh-pages:{path}")


def _versions(repo: Path) -> list[dict]:
    return json.loads(_branch_file(repo, "versions.json"))


def test_a_first_publish(repo: Path, tmp_path: Path):
    assert publish_version.publish(repo, _site(tmp_path, "a", PAGES), "dev", push=True)
    _git(repo, "fetch", "-q", "origin")
    assert _versions(repo) == [{"version": "dev", "title": "dev", "aliases": []}]
    assert _branch_file(repo, "dev/reference/isa/instructions/index.html") == "<h1>isa</h1>"
    assert 'url=dev/"' in _branch_file(repo, "index.html")
    assert _branch_file(repo, ".nojekyll") == ""


def test_a_republish_with_no_change_changes_nothing(repo: Path, tmp_path: Path):
    site = _site(tmp_path, "a", PAGES)
    assert publish_version.publish(repo, site, "dev", push=True)
    _git(repo, "fetch", "-q", "origin")
    head = _git(repo, "rev-parse", "origin/gh-pages")
    assert not publish_version.publish(repo, site, "dev", push=True)
    _git(repo, "fetch", "-q", "origin")
    assert _git(repo, "rev-parse", "origin/gh-pages") == head


def test_a_master_publish_changes_only_dev(repo: Path, tmp_path: Path):
    publish_version.publish(repo, _site(tmp_path, "v1", PAGES), "v1.0", ["latest"], push=True)
    publish_version.publish(repo, _site(tmp_path, "a", PAGES), "dev", push=True)
    _git(repo, "fetch", "-q", "origin")
    before = _git(repo, "rev-parse", "origin/gh-pages").strip()
    changed = {**PAGES, "index.html": "<h1>home, changed</h1>"}
    publish_version.publish(repo, _site(tmp_path, "b", changed), "dev", push=True)
    _git(repo, "fetch", "-q", "origin")
    files = _git(repo, "diff", "--name-only", before, "origin/gh-pages").split()
    assert files == ["dev/index.html"]


def test_an_alias_moves(repo: Path, tmp_path: Path):
    publish_version.publish(repo, _site(tmp_path, "v1", PAGES), "v1.0", ["latest"], push=True)
    publish_version.publish(repo, _site(tmp_path, "a", PAGES), "dev", push=True)
    publish_version.publish(repo, _site(tmp_path, "v2", PAGES), "v1.1", ["latest"], push=True)
    _git(repo, "fetch", "-q", "origin")
    assert _versions(repo) == [
        {"version": "dev", "title": "dev", "aliases": []},
        {"version": "v1.1", "title": "v1.1", "aliases": ["latest"]},
        {"version": "v1.0", "title": "v1.0", "aliases": []},
    ]
    # mike's redirect pages: the alias's index goes to the version's directory.
    assert 'url=../v1.1/"' in _branch_file(repo, "latest/index.html")
    nested = _branch_file(repo, "latest/reference/isa/instructions/index.html")
    assert 'url=../../../../v1.1/reference/isa/instructions/"' in nested
    assert 'url=latest/"' in _branch_file(repo, "index.html")


def test_versions_json_is_what_mike_reads(repo: Path, tmp_path: Path):
    publish_version.publish(repo, _site(tmp_path, "v1", PAGES), "v1.0", ["latest"], push=True)
    publish_version.publish(repo, _site(tmp_path, "a", PAGES), "dev", push=True)
    publish_version.publish(repo, _site(tmp_path, "v2", PAGES), "v1.1", push=True)
    _git(repo, "fetch", "-q", "origin")
    ours = _versions(repo)
    versions = Versions.loads(json.dumps(ours))
    # mike lists versions in the same order, with the same aliases.
    assert [str(info.version) for info in versions] == [entry["version"] for entry in ours]
    assert versions.find("latest") == ("v1.0", "latest")


def test_bad_versions_are_refused(repo: Path, tmp_path: Path):
    site = _site(tmp_path, "a", PAGES)
    with pytest.raises(ValueError, match="dev or vX.Y"):
        publish_version.publish(repo, site, "release-1", push=True)
    with pytest.raises(ValueError, match="cannot alias itself"):
        publish_version.publish(repo, site, "v1.0", ["v1.0"], push=True)


def test_the_cli_refuses_to_run_without_repo(tmp_path: Path, capsys):
    site = _site(tmp_path, "a", PAGES)
    with pytest.raises(SystemExit) as error:
        publish_version.main(["--site", str(site), "--version", "dev"])
    assert error.value.code == 2
    assert "--repo" in capsys.readouterr().err


def test_the_cli_commits_locally_unless_told_to_push(repo: Path, tmp_path: Path):
    site = _site(tmp_path, "a", PAGES)
    assert publish_version.main(["--repo", str(repo), "--site", str(site), "--version", "dev"]) == 0
    assert _git(repo, "rev-parse", "--verify", "--quiet", "refs/heads/gh-pages")
    assert _git(repo, "ls-remote", "origin", "gh-pages") == ""
    publish_version.main(["--repo", str(repo), "--site", str(site), "--version", "dev", "--push"])
    assert _git(repo, "ls-remote", "origin", "gh-pages") != ""


def test_the_root_redirects_to_dev_until_a_release_exists(repo: Path, tmp_path: Path):
    publish_version.publish(repo, _site(tmp_path, "a", PAGES), "dev", push=True)
    _git(repo, "fetch", "-q", "origin")
    assert 'url=dev/"' in _branch_file(repo, "index.html")
    publish_version.publish(repo, _site(tmp_path, "v1", PAGES), "v1.0", ["latest"], push=True)
    _git(repo, "fetch", "-q", "origin")
    assert 'url=latest/"' in _branch_file(repo, "index.html")


def test_a_tag_publish_leaves_dev_alone(repo: Path, tmp_path: Path):
    publish_version.publish(repo, _site(tmp_path, "a", PAGES), "dev", push=True)
    _git(repo, "fetch", "-q", "origin")
    before = _git(repo, "rev-parse", "origin/gh-pages").strip()
    publish_version.publish(repo, _site(tmp_path, "v1", PAGES), "v1.0", ["latest"], push=True)
    _git(repo, "fetch", "-q", "origin")
    files = _git(repo, "diff", "--name-only", before, "origin/gh-pages").split()
    assert files and not [f for f in files if f.startswith("dev/")]


def test_latest_never_moves_to_an_older_release(repo: Path, tmp_path: Path):
    publish_version.publish(repo, _site(tmp_path, "v2", PAGES), "v2.0", ["latest"], push=True)
    publish_version.publish(repo, _site(tmp_path, "v1", PAGES), "v1.3", ["latest"], push=True)
    _git(repo, "fetch", "-q", "origin")
    assert _versions(repo) == [
        {"version": "v2.0", "title": "v2.0", "aliases": ["latest"]},
        {"version": "v1.3", "title": "v1.3", "aliases": []},
    ]


@pytest.mark.parametrize("alias", ["", ".", "..", "a/b", "dev", "v1.0", "v2.0"])
def test_invalid_aliases_are_refused(repo: Path, tmp_path: Path, alias: str):
    publish_version.publish(repo, _site(tmp_path, "v1", PAGES), "v1.0", push=True)
    with pytest.raises(ValueError):
        publish_version.publish(repo, _site(tmp_path, "v2", PAGES), "v2.0", [alias], push=True)


def test_nothing_is_left_behind_after_an_error(repo: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        publish_version.publish(repo, tmp_path / "missing.tar", "dev", push=True)
    assert len(_git(repo, "worktree", "list").splitlines()) == 1
    assert "publish-version" not in _git(repo, "branch", "--list")
    assert _git(repo, "ls-remote", "origin", "gh-pages") == ""


def test_a_push_that_loses_a_race_is_redone_on_top(repo: Path, tmp_path: Path, monkeypatch):
    """A tag publish and a master publish race: the loser rebuilds on the winner."""
    publish_version.publish(repo, _site(tmp_path, "a", PAGES), "dev", push=True)
    rival = tmp_path / "rival"
    subprocess.run(
        ["git", "clone", "-q", _git(repo, "remote", "get-url", "origin").strip(), str(rival)],
        check=True,
    )
    for key, value in (("user.name", "rival"), ("user.email", "rival@example.com")):
        _git(rival, "config", key, value)
    real_git = publish_version._git
    raced = []

    def racing_git(where, *args, check=True):
        if args[:1] == ("push",) and not raced:
            raced.append(True)  # the rival lands first, once
            publish_version.publish(
                rival, _site(tmp_path, "v1", PAGES), "v1.0", ["latest"], push=True
            )
        return real_git(where, *args, check=check)

    monkeypatch.setattr(publish_version, "_git", racing_git)
    changed = {**PAGES, "index.html": "<h1>dev, changed</h1>"}
    assert publish_version.publish(repo, _site(tmp_path, "b", changed), "dev", push=True)
    _git(repo, "fetch", "-q", "origin")
    assert [e["version"] for e in _versions(repo)] == ["dev", "v1.0"]
    assert _branch_file(repo, "dev/index.html") == "<h1>dev, changed</h1>"
