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

import json
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


# -- gate_verdict derivation (2026-09-03 false-REJECT regression) ----------
#
# Symptom: a gate whose run summary OPENS 'CDP FINAL GATE ACCEPT - ...' was
# derived REJECTED because the old code substring-scanned the ENTIRE
# concatenated (summary + result) text with reject markers checked first, so
# reviewer history in the prose ('Security REJECT -> remediation') flipped the
# verdict. Fix (CDP decision 2026-09-03): task.result parsed FIRST in
# isolation; free-text fallback limited to the opening verdict window
# (GATE_OPENING_WINDOW) with gate-scoped phrase priority.


def _run_gate_mission(conn, *, result=None, summary=None) -> tuple[str, str]:
    """Create + finish a canonical mission (umbrella -> done gate).

    ``result`` defaults to None so tests can model the real regression case
    (gate completed WITHOUT an explicit structured task.result).
    """
    umbrella = kb.create_task(conn, title="Mission", role="umbrella")
    gate = kb.create_task(conn, title="Gate", role="gate", parents=[umbrella])
    kb.claim_task(conn, umbrella)
    kb.complete_task(conn, umbrella, result="mission underway")
    assert kb.claim_task(conn, gate) is not None
    ok = kb.complete_task(conn, gate, result=result, summary=summary)
    assert ok
    return umbrella, gate


def test_gate_verdict_accept_survives_reviewer_history_in_summary(conn):
    """Regression: 'Security REJECT' prose later in the summary must not flip
    a gate whose summary opens with its own ACCEPT verdict line."""
    umbrella, gate = _run_gate_mission(
        conn,
        result=None,
        summary=(
            "CDP FINAL GATE ACCEPT - AppStock regression stabilization accepted. "
            "Evidence: Product register, UX audit, Lead implementation, "
            "Security REJECT->remediation, DevOps PASS, QA ACCEPT verified."
        ),
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.gate_verdict(conn, gate_task) is True


def test_gate_verdict_structured_result_parsed_in_isolation(conn):
    """result=ACCEPTED wins even when the summary mentions a REJECT history."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="ACCEPTED",
        summary="Security REJECT -> remediation, then final acceptance; all checks green.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.gate_verdict(conn, gate_task) is True


def test_gate_verdict_reject_result_wins_over_accept_prose(conn):
    """result=REJECTED stays rejected regardless of accept prose in summary."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="REJECTED",
        summary="QA ACCEPTED on first pass; blockers remain open.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.gate_verdict(conn, gate_task) is False


def test_gate_verdict_gate_scoped_phrase_beats_earlier_plain_marker(conn):
    """A gate's OWN phrase ('GATE ACCEPT') outranks an earlier plain REJECT
    marker from reviewer prose inside the opening window."""
    umbrella, gate = _run_gate_mission(
        conn,
        result=None,
        summary="REJECT noted by QA; however CDP FINAL GATE ACCEPT - final.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.gate_verdict(conn, gate_task) is True


def test_gate_verdict_marker_outside_opening_window_ignored(conn):
    """Free-text fallback only reads the opening verdict statement; a REJECT
    buried later in a neutral summary is not a verdict (fail toward human
    check instead of a wrong derivation)."""
    umbrella, gate = _run_gate_mission(
        conn,
        result=None,
        summary=("Neutral narrative text " * 10)
        + "FINAL VERDICT REJECTED - changes required.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.gate_verdict(conn, gate_task) is None


def test_verdict_from_marker_text_boundaries():
    """Helper boundaries: empty/None text is not a verdict; single-token
    structured results resolve unambiguously."""
    assert kb._verdict_from_marker_text(None) is None
    assert kb._verdict_from_marker_text("") is None
    assert kb._verdict_from_marker_text("ACCEPTED") is True
    assert kb._verdict_from_marker_text("REJECTED") is False
    assert kb._verdict_from_marker_text("NOT ACCEPTED") is False


# -- recompute: corrective RECOMPUTED handoff -------------------------------


def _handoff_comment_bodies(conn, task_id) -> list[str]:
    return [
        c.body for c in kb.list_comments(conn, task_id) if kb.HANDOFF_MARKER in c.body
    ]


def _newest_handoff_event_payload(conn, task_id) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, kb.HANDOFF_EVENT_KIND),
    ).fetchone()
    assert row is not None and row["payload"]
    return json.loads(row["payload"])


def test_recompute_appends_corrective_handoff_when_verdict_changes(conn):
    """Stored REJECTED handoff + structured result later corrected to ACCEPTED
    -> recompute appends a RECOMPUTED handoff; the original stays for audit."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="REJECT",
        summary="Gate rejected: blockers open.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(conn, gate_task) is True
    assert len(_handoff_comment_bodies(conn, umbrella)) == 1
    assert _newest_handoff_event_payload(conn, umbrella)["verdict"].startswith("REJECT")

    # The record is corrected: structured task.result now carries ACCEPTED.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET result = ? WHERE id = ?", ("ACCEPTED", gate))

    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None and gate_task.result == "ACCEPTED"
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is True

    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 2  # original preserved + corrective appended
    assert "RECOMPUTED" in bodies[1]
    newest = _newest_handoff_event_payload(conn, umbrella)
    assert newest["verdict"] == "ACCEPTED"
    assert newest["recomputed"] is True

    # Derivation now matches the stored verdict -> further recomputes no-op.
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is False
    assert len(_handoff_comment_bodies(conn, umbrella)) == 2


def test_recompute_noop_when_verdict_unchanged(conn):
    """Stored ACCEPTED + derived ACCEPTED -> recompute never re-emits."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="ACCEPTED",
        summary="All green.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(conn, gate_task) is True
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is False
    assert len(_handoff_comment_bodies(conn, umbrella)) == 1


def test_recompute_without_prior_handoff_emits_normal(conn):
    """recompute=True before any handoff exists behaves like a normal first
    emission (no RECOMPUTED marking, no corrective event)."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="ACCEPTED",
        summary="All green.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is True
    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 1
    assert "RECOMPUTED" not in bodies[0]
    assert _newest_handoff_event_payload(conn, umbrella)["recomputed"] is False


# -- recompute: next-action TYPE refresh (2026-09-03 delivery sessions) ------
#
# The recompute path must refresh the persisted handoff when the RESOLVED
# NEXT-ACTION TYPE moves even though the verdict itself is unchanged (repo
# state advanced, e.g. dirty -> committed -> pushed). Before the fix the
# guard compared only the verdict, so a stale handoff kept proposing
# DELIVERY_CHECKPOINT after the work was already committed.


def _next_action_type(payload: dict) -> str | None:
    na = payload.get("next_action") or payload.get("decision") or {}
    return na.get("type") or na.get("actionType")


def _commit_all(repo) -> None:
    import subprocess

    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "advance"],
        check=True,
    )


def test_recompute_refreshes_handoff_when_next_action_type_moves(conn, tmp_path):
    """Stored DELIVERY_CHECKPOINT + repo advanced dirty -> committed yields a
    RECOMPUTED handoff carrying PUSH_CHECKPOINT; further recomputes no-op."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    (repo / "a.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    (repo / "b.txt").write_text("y")  # dirty at first emission

    umbrella = kb.create_task(conn, title="Mission", role="umbrella")
    gate = kb.create_task(
        conn, title="Gate", role="gate", parents=[umbrella],
        workspace_kind="dir", workspace_path=str(repo),
    )
    kb.claim_task(conn, umbrella)
    kb.complete_task(conn, umbrella, result="mission underway")
    assert kb.claim_task(conn, gate) is not None
    assert kb.complete_task(conn, gate, result="ACCEPTED", summary="All green.")
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None

    # First emission while the tree is dirty -> DELIVERY_CHECKPOINT stored.
    assert kb.emit_terminal_handoff(conn, gate_task) is True
    first = _newest_handoff_event_payload(conn, umbrella)
    assert first["verdict"] == "ACCEPTED"
    assert _next_action_type(first) == "DELIVERY_CHECKPOINT"

    # Repo advances (dirty -> committed). Verdict unchanged, type must move.
    _commit_all(repo)
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is True

    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 2  # original preserved + type-refresh appended
    assert "RECOMPUTED" in bodies[1]
    newest = _newest_handoff_event_payload(conn, umbrella)
    assert newest["verdict"] == "ACCEPTED"
    assert _next_action_type(newest) == "PUSH_CHECKPOINT"
    assert newest["recomputed"] is True

    # Stored type now matches the derived type -> further recomputes no-op.
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is False
    assert len(_handoff_comment_bodies(conn, umbrella)) == 2


def test_recompute_type_refresh_keeps_verdict_when_type_unchanged(conn, tmp_path):
    """A repo-advance with no type move must never fabricate a corrective
    handoff: verdict ACCEPTED + already-clean committed state stays PUSH
    (single emission, recompute no-op)."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    (repo / "a.txt").write_text("x")
    _commit_all(repo)  # clean committed (unpushed) from the start

    umbrella = kb.create_task(conn, title="Mission", role="umbrella")
    gate = kb.create_task(
        conn, title="Gate", role="gate", parents=[umbrella],
        workspace_kind="dir", workspace_path=str(repo),
    )
    kb.claim_task(conn, umbrella)
    kb.complete_task(conn, umbrella, result="mission underway")
    assert kb.claim_task(conn, gate) is not None
    assert kb.complete_task(conn, gate, result="ACCEPTED", summary="All green.")
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None

    assert kb.emit_terminal_handoff(conn, gate_task) is True
    assert _next_action_type(_newest_handoff_event_payload(conn, umbrella)) == (
        "PUSH_CHECKPOINT"
    )
    # No state moved (still clean + committed + unpushed): recompute no-ops.
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is False
    assert len(_handoff_comment_bodies(conn, umbrella)) == 1
