"""Tests for the terminal mission handoff (post-gate lifecycle).

Covers the canonical lifecycle distinction:

    MISSION EXECUTION COMPLETED  (gate card -> done)
    vs
    MISSION LIFECYCLE HANDOFF COMPLETED  (terminal handoff emitted once)

Spec scenarios:
  A. final gate ACCEPTED + dirty tree  -> handoff emitted, delivery proposed,
     approval requested (repo_state.dirty=True and handoff event present)
  B. final gate ACCEPTED + already committed/pushed -> handoff still emitted
     (repo facts clean) with no redundant re-commit proposal (resolver-level)
  C. QA rejected (verdict False)      -> verdict recorded as rejected (the
     resolver maps that to remediation)
  D. duplicate UI scope already covered -> no duplicate follow-up card is
     created by the handoff machinery (the handoff never creates cards)
  E. restart/replay                    -> handoff not emitted twice
  F. user approval pending             -> state survives restart (handoff is
     persisted on the board, so a fresh connection observes it)
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _run_mission(conn, *, verdict_result: str, verdict_summary: str) -> tuple[str, str]:
    """Create + run a canonical mission: umbrella -> gate -> done gate."""
    umbrella = kb.create_task(conn, title="Test mission", role="umbrella")
    gate = kb.create_task(conn, title="Final gate", role="gate", parents=[umbrella])
    kb.claim_task(conn, umbrella)
    kb.complete_task(conn, umbrella, result="mission underway")
    assert kb.claim_task(conn, gate) is not None
    ok = kb.complete_task(
        conn, gate,
        result=verdict_result,
        summary=verdict_summary,
    )
    assert ok
    return umbrella, gate


def test_a_accepted_gate_emits_handoff_exactly_once(conn):
    """Final gate ACCEPTED -> terminal handoff emitted on the umbrella once."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="Independent QA accepted; all checks green.",
    )
    emitted = kb.emit_terminal_handoffs_if_due(conn)
    assert emitted == [gate]

    comments = kb.list_comments(conn, umbrella)
    handoff_comments = [c for c in comments if kb.HANDOFF_MARKER in c.body]
    assert len(handoff_comments) == 1
    body = handoff_comments[0].body
    assert "Final gate" in body
    assert "ACCEPTED" in body

    # Handoff event recorded for gateway notifiers.
    events = kb.list_events(conn, umbrella)
    assert any(e.kind == kb.HANDOFF_EVENT_KIND for e in events)

    # D: the handoff never fabricates follow-up cards.
    all_tasks = conn.execute("SELECT id FROM tasks").fetchall()
    assert len(all_tasks) == 2  # umbrella + gate — nothing extra created


def test_b_no_duplicate_emission_on_replay(conn):
    """Restart/replay: second scan emits nothing (E), state persisted (F)."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All green.",
    )
    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]

    # Simulate a restarted session: brand-new connection (fresh observer).
    with kb.connect() as conn2:
        assert kb.emit_terminal_handoffs_if_due(conn2) == []
        comments = kb.list_comments(conn2, umbrella)
        assert sum(1 for c in comments if kb.HANDOFF_MARKER in c.body) == 1


def test_c_rejected_gate_records_reject(conn):
    """QA rejected -> verdict False on the synthesized handoff."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="Independent QA rejected; FAIL-1/2/3 open.",
    )
    gate_task = kb.get_task(conn, gate)
    umbrella_task = kb.get_task(conn, umbrella)
    assert gate_task is not None and umbrella_task is not None
    snapshot = kb.synthesize_terminal_handoff(conn, gate_task, umbrella_task)
    assert snapshot["verdict"] is False
    kb.emit_terminal_handoffs_if_due(conn)
    comments = kb.list_comments(conn, umbrella)
    assert any("REJECTED" in c.body for c in comments if kb.HANDOFF_MARKER in c.body)


def test_d_non_gate_completion_never_emits(conn):
    """Ordinary (non role=gate) completion emits nothing."""
    plain = kb.create_task(conn, title="Plain task")
    kb.claim_task(conn, plain)
    kb.complete_task(conn, plain, result="done")
    assert kb.emit_terminal_handoffs_if_due(conn) == []
    comments = kb.list_comments(conn, plain)
    assert not any(kb.HANDOFF_MARKER in c.body for c in comments)


def test_repo_state_reflects_git_dirty(kanban_home, conn, tmp_path):
    """repo_state detects a dirty tree in a dir workspace (A)."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    (repo / "a.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    (repo / "b.txt").write_text("y")  # dirty

    gate = kb.create_task(
        conn, title="gate", role="gate",
        workspace_kind="dir", workspace_path=str(repo),
    )
    state = kb.repo_state_for(kb.get_task(conn, gate))
    assert state.get("dirty") is True
    assert state.get("committed") is True
    assert state.get("branch")


def test_find_umbrella_and_unresolved_blocker(conn):
    """Umbrella discovery prefers role=umbrella; blockers are reported."""
    umbrella = kb.create_task(conn, title="Mission", role="umbrella")
    gate = kb.create_task(conn, title="Gate", role="gate", parents=[umbrella])
    kb.claim_task(conn, umbrella)
    kb.complete_task(conn, umbrella, result="underway")

    blocker = kb.create_task(conn, title="Blocked child", parents=[umbrella])
    kb.block_task(conn, blocker, reason="needs input", kind="needs_input")

    found = kb.find_mission_parent(conn, gate)
    assert found is not None and found.id == umbrella
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    snapshot = kb.synthesize_terminal_handoff(conn, gate_task, kb.get_task(conn, umbrella))
    assert blocker in snapshot["blockers"]


# -- resolver (pure) -----------------------------------------------------


def test_resolver_dirty_tree_proposes_delivery_with_approval():
    """Accepted + dirty tree -> DELIVERY_CHECKPOINT, approval required (A)."""
    action = kb.resolve_next_action({
        "verdict": True, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": True, "committed": True, "pushed": False,
                       "branch": "main", "unpushed_commits": 1},
    })
    assert action["type"] == "DELIVERY_CHECKPOINT"
    assert action["requiresApproval"] is True
    assert "uncommitted" in action["reason"]


def test_resolver_committed_unpushed_proposes_push():
    """Accepted + committed not pushed -> PUSH_CHECKPOINT, no redundant commit."""
    action = kb.resolve_next_action({
        "verdict": True, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": False, "committed": True, "pushed": False,
                       "branch": "feat/x", "unpushed_commits": 2},
    })
    assert action["type"] == "PUSH_CHECKPOINT"
    assert any("commit" not in s.lower() or "no separate" in s.lower() for s in action["sequence"])


def test_resolver_committed_pushed_proposes_integration():
    """Accepted + committed + pushed -> INTEGRATION_REVIEW (B)."""
    action = kb.resolve_next_action({
        "verdict": True, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": False, "committed": True, "pushed": True,
                       "branch": "main", "unpushed_commits": 0},
    })
    assert action["type"] == "INTEGRATION_REVIEW"
    assert action["requiresApproval"] is True  # merge approval


def test_resolver_rejected_proposes_remediation():
    """QA rejected -> REMEDIATION, never delivery (C)."""
    action = kb.resolve_next_action({
        "verdict": False, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": True, "committed": False, "pushed": False},
    })
    assert action["type"] == "REMEDIATION"
    assert action["requiresApproval"] is False


def test_resolver_blocker_requires_decision():
    """Unresolved blocker -> AWAITING_DECISION before any workflow."""
    action = kb.resolve_next_action({
        "verdict": True, "active_workers": [],
        "blockers": ["t_blocked1"],
        "repo_state": {"dirty": True, "committed": False, "pushed": False},
    })
    assert action["type"] == "AWAITING_DECISION"
    assert action["requiresApproval"] is True


def test_resolver_active_worker_waits():
    """Active workers -> AWAITING_WORKERS; no premature delivery proposal."""
    action = kb.resolve_next_action({
        "verdict": True, "active_workers": ["t_run (hotelos-lead)"],
        "blockers": [], "repo_state": {"dirty": True},
    })
    assert action["type"] == "AWAITING_WORKERS"
