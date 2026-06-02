"""Tests for the `memory dismiss` CLI command (bulk event dismissal).

Uses a fake Qdrant client (same spirit as MockEmbedder elsewhere) so the
command's client-side logic — keep-exclusion, dry-run vs apply, batching — is
covered without a live Qdrant. The server-side source/extracted filter is
Qdrant's contract; the fake returns an already-filtered candidate set.
"""

from click.testing import CliRunner

import entity_memory.cli as cli_mod
from entity_memory.cli import main


class _FakePoint:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


class _FakeClient:
    def __init__(self, points):
        self._points = points
        self.set_payload_calls = []

    def collection_exists(self, name):
        return True

    def scroll(self, collection_name, scroll_filter=None, limit=100,
               offset=None, with_payload=True):
        # Single page; real filtering is Qdrant's job (see module docstring).
        return list(self._points), None

    def set_payload(self, collection_name, payload, points, wait=False):
        self.set_payload_calls.append(
            {"collection": collection_name, "payload": payload,
             "points": list(points), "wait": wait}
        )


def _candidates(n):
    return [
        _FakePoint(
            f"id-{i}",
            {"text": f"event {i}", "source": "claude_ai_import_conversation", "extracted": False},
        )
        for i in range(n)
    ]


def test_dry_run_reports_count_and_writes_nothing(monkeypatch):
    fake = _FakeClient(_candidates(3))
    monkeypatch.setattr(cli_mod, "get_client", lambda: fake)

    result = CliRunner().invoke(
        main, ["dismiss", "--source", "claude_ai_import_conversation"]
    )

    assert result.exit_code == 0, result.output
    assert "Matched 3 unextracted events" in result.output
    assert "DRY-RUN" in result.output
    assert fake.set_payload_calls == []  # dry-run must not write


def test_apply_marks_all_matched_extracted(monkeypatch):
    fake = _FakeClient(_candidates(3))
    monkeypatch.setattr(cli_mod, "get_client", lambda: fake)

    result = CliRunner().invoke(
        main, ["dismiss", "--source", "claude_ai_import_conversation", "--apply"]
    )

    assert result.exit_code == 0, result.output
    assert "Dismissed 3 events" in result.output
    assert len(fake.set_payload_calls) == 1
    call = fake.set_payload_calls[0]
    assert call["payload"] == {"extracted": True}
    assert set(call["points"]) == {"id-0", "id-1", "id-2"}
    assert call["wait"] is True  # one-shot CLI mutation must be durable on return


def test_keep_excludes_listed_ids(monkeypatch):
    fake = _FakeClient(_candidates(3))
    monkeypatch.setattr(cli_mod, "get_client", lambda: fake)

    result = CliRunner().invoke(
        main,
        ["dismiss", "--source", "claude_ai_import_conversation", "--keep", "id-1", "--apply"],
    )

    assert result.exit_code == 0, result.output
    assert "Dismissed 2 events" in result.output
    call = fake.set_payload_calls[0]
    assert set(call["points"]) == {"id-0", "id-2"}
    assert "id-1" not in call["points"]
