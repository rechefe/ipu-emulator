"""Tests for docs/update_release_log.py (no network access)."""

import socket
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from docs import update_release_log as rl


def _pr(number, title, merged_at, updated_at=None):
    return {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/{rl.REPO}/pull/{number}",
        "merged_at": merged_at,
        "updated_at": updated_at or merged_at,
    }


_NETWORK_ERRORS = [
    URLError("Name or service not known"),
    TimeoutError("timed out"),
    socket.timeout("read timed out"),
    HTTPError("https://api.github.com", 403, "rate limited", Message(), None),
    ConnectionResetError("reset by peer"),
]


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


def _raise(exc):
    def fake_get(url, token):
        raise exc

    return fake_get


@pytest.mark.parametrize("exc", _NETWORK_ERRORS, ids=lambda e: type(e).__name__)
def test_network_error_writes_minimal_log(monkeypatch, tmp_path, exc):
    monkeypatch.setattr(rl, "_get", _raise(exc))
    out = tmp_path / "release-log.md"
    assert rl.main(["--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith(rl.HEADER)
    assert rl.parse_rows(text) == {}
    assert "*Last updated:" in text


@pytest.mark.parametrize("exc", _NETWORK_ERRORS, ids=lambda e: type(e).__name__)
def test_latest_network_error_without_output_writes_minimal_log(monkeypatch, tmp_path, exc):
    monkeypatch.setattr(rl, "_get", _raise(exc))
    out = tmp_path / "release-log.md"
    assert rl.main(["--latest", "--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith(rl.HEADER)


@pytest.mark.parametrize("exc", _NETWORK_ERRORS, ids=lambda e: type(e).__name__)
def test_latest_network_error_leaves_existing_file_unchanged(monkeypatch, tmp_path, exc):
    out = tmp_path / "release-log.md"
    out.write_text(
        rl.build_content([_pr(1, "First", "2024-01-01T00:00:00Z")]) + "trailing bytes",
        encoding="utf-8",
    )
    before = out.read_bytes()
    monkeypatch.setattr(rl, "_get", _raise(exc))
    assert rl.main(["--latest", "--output", str(out)]) == 0
    assert out.read_bytes() == before


def _three_row_log():
    return rl.build_content(
        [
            _pr(30, "Third | with pipe", "2024-03-01T12:00:00Z"),
            _pr(20, "Second", "2024-02-01T12:00:00Z"),
            _pr(10, "First", "2024-01-01T12:00:00Z"),
        ]
    )


def test_parse_rows_round_trips_format_row():
    rows = rl.parse_rows(_three_row_log())
    assert sorted(rows) == [10, 20, 30]
    assert rows[30] == (
        "2024-03-01 12:00",
        rl.format_row(_pr(30, "Third | with pipe", "2024-03-01T12:00:00Z")),
    )


def test_merge_rows_adds_new_pr_newest_first():
    merged = rl.merge_rows(_three_row_log(), [_pr(40, "Fourth", "2024-04-01T12:00:00Z")])
    rows = rl.parse_rows(merged)
    assert len(rows) == 4
    numbers = [int(line.split("[#")[1].split("]")[0]) for line in merged.splitlines() if line.startswith("| [#")]
    assert numbers == [40, 30, 20, 10]
    assert merged.startswith(rl.HEADER)


def test_merge_rows_sorts_by_merge_time_not_insertion():
    merged = rl.merge_rows(_three_row_log(), [_pr(15, "Late number, mid date", "2024-01-15T00:00:00Z")])
    numbers = [int(line.split("[#")[1].split("]")[0]) for line in merged.splitlines() if line.startswith("| [#")]
    assert numbers == [30, 20, 15, 10]


def test_merge_rows_replaces_same_number():
    merged = rl.merge_rows(_three_row_log(), [_pr(20, "Second, retitled", "2024-02-01T12:00:00Z")])
    rows = rl.parse_rows(merged)
    assert len(rows) == 3
    assert "Second, retitled" in rows[20][1]
    assert "| Second |" not in merged


def test_merge_rows_on_empty_log_gives_one_row():
    merged = rl.merge_rows("", [_pr(5, "Only", "2024-01-01T00:00:00Z")])
    assert merged.startswith(rl.HEADER)
    assert list(rl.parse_rows(merged)) == [5]


def _page_ordered_by_updated():
    # Ordered by updated_at desc; the most recently *merged* PR is not first.
    return [
        _pr(7, "Updated last, merged early", "2024-01-01T00:00:00Z", "2024-06-01T00:00:00Z"),
        {"number": 8, "title": "Closed unmerged", "html_url": "x", "merged_at": None},
        _pr(9, "Merged last", "2024-05-01T00:00:00Z", "2024-05-02T00:00:00Z"),
        _pr(6, "Merged mid", "2024-03-01T00:00:00Z", "2024-04-01T00:00:00Z"),
    ]


def test_latest_picks_greatest_merged_at(monkeypatch):
    calls = []

    def fake_get(url, token):
        calls.append(url)
        return _page_ordered_by_updated()

    monkeypatch.setattr(rl, "_get", fake_get)
    prs = rl.fetch_merged_prs(None, latest_only=True)
    assert [pr["number"] for pr in prs] == [9]
    assert len(calls) == 1


def test_latest_main_merges_into_existing_file(monkeypatch, tmp_path):
    out = tmp_path / "release-log.md"
    out.write_text(_three_row_log(), encoding="utf-8")
    monkeypatch.setattr(rl, "_get", lambda url, token: _page_ordered_by_updated())
    assert rl.main(["--latest", "--output", str(out)]) == 0
    rows = rl.parse_rows(out.read_text(encoding="utf-8"))
    assert sorted(rows) == [9, 10, 20, 30]


def test_latest_main_without_output_writes_one_row(monkeypatch, tmp_path):
    out = tmp_path / "release-log.md"
    monkeypatch.setattr(rl, "_get", lambda url, token: _page_ordered_by_updated())
    assert rl.main(["--latest", "--output", str(out)]) == 0
    assert list(rl.parse_rows(out.read_text(encoding="utf-8"))) == [9]


def test_full_refresh_pages_and_sorts(monkeypatch, tmp_path):
    page1 = [_pr(n, f"PR {n}", f"2024-01-01T00:{n % 60:02d}:00Z") for n in range(100)]
    page2 = [_pr(200, "Newest", "2024-02-01T00:00:00Z")]

    def fake_get(url, token):
        return page1 if "page=1&" in url else page2

    monkeypatch.setattr(rl, "_get", fake_get)
    out = tmp_path / "release-log.md"
    assert rl.main(["--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert len(rl.parse_rows(text)) == 101
    first_row = next(line for line in text.splitlines() if line.startswith("| [#"))
    assert first_row.startswith("| [#200]")
