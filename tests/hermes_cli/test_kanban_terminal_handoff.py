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
    assert state.get("committed") is False
    assert state.get("branch")


# -- repo git truth (2026-09-04 delivery-checkpoint-execution-git-truth) --
#
# repo_state_for must report Git truth from actual commands: FULL head SHA,
# dirty covering tracked + staged + untracked, and pushed True ONLY when a
# remote-tracking ref exists whose SHA equals the local HEAD SHA. Missing or
# ambiguous remote evidence must never yield pushed=True.


def _repo_git(repo, *args):
    import subprocess

    subprocess.run(["git", "-C", str(repo), *args], check=True)


def _init_repo(repo, branch="main"):
    import subprocess

    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", branch, str(repo)], check=True)
    _repo_git(repo, "config", "user.email", "t@t")
    _repo_git(repo, "config", "user.name", "T")
    return repo


def _commit_file(repo, name="a.txt", content="x", msg="commit"):
    (repo / name).write_text(content)
    _repo_git(repo, "add", "-A")
    _repo_git(repo, "commit", "-qm", msg)
    return repo


def _bare_remote(tmp_path, name):
    import subprocess

    remote = tmp_path / name
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return remote


def _gate_on_repo(conn, repo):
    return kb.create_task(
        conn, title="gate", role="gate",
        workspace_kind="dir", workspace_path=str(repo),
    )


def test_repo_state_pushed_false_when_remote_ahead_of_local(conn, tmp_path):
    """Git truth: pushed requires remote SHA == local SHA. A local branch
    that is BEHIND its remote (the remote advanced elsewhere) is NOT pushed —
    even though the local branch has no unpushed commits of its own."""
    origin = _bare_remote(tmp_path, "origin.git")
    local = _init_repo(tmp_path / "local")
    _repo_git(local, "remote", "add", "origin", str(origin))
    _commit_file(local)
    _repo_git(local, "push", "-q", "-u", "origin", "main")

    gate = _gate_on_repo(conn, local)
    state = kb.repo_state_for(kb.get_task(conn, gate))
    assert state.get("pushed") is True
    assert state.get("remote_head") == state.get("head")
    assert state.get("ahead") == 0
    assert state.get("behind") == 0

    # The remote advances independently (a second clone commits + pushes).
    other = _init_repo(tmp_path / "other")
    _repo_git(other, "remote", "add", "origin", str(origin))
    _repo_git(other, "fetch", "-q", "origin")
    _repo_git(other, "checkout", "-q", "-b", "main", "origin/main")
    _commit_file(other, "b.txt", "y", "remote advance")
    _repo_git(other, "push", "-q", "origin", "main")

    # Refresh the local view of the remote before re-probing (remote-tracking
    # refs only move on fetch — the probe reports the known remote truth).
    _repo_git(local, "fetch", "-q", "origin")

    state2 = kb.repo_state_for(kb.get_task(conn, gate))
    assert state2.get("committed") is True
    assert state2.get("remote_head") != state2.get("head")
    assert state2.get("behind", 0) >= 1
    assert state2.get("pushed") is False  # local != remote -> never true


def test_repo_state_no_or_ambiguous_remote_never_pushed(conn, tmp_path):
    """Missing/ambiguous remote => pushed False/unknown, never True."""
    local = _init_repo(tmp_path / "local")
    _commit_file(local)
    gate = _gate_on_repo(conn, local)
    state = kb.repo_state_for(kb.get_task(conn, gate))
    assert state.get("committed") is True
    assert state.get("pushed") is False
    assert state.get("remote_head") is None

    # Two remotes track the same branch name at DIFFERENT SHAs: the remote
    # truth is ambiguous -> pushed must not be reported as True.
    origin = _bare_remote(tmp_path, "origin.git")
    fork = _bare_remote(tmp_path, "fork.git")
    _repo_git(local, "remote", "add", "origin", str(origin))
    _repo_git(local, "remote", "add", "fork", str(fork))
    _repo_git(local, "push", "-q", "-u", "origin", "main")
    _repo_git(local, "push", "-q", "fork", "main")
    other = _init_repo(tmp_path / "other")
    _repo_git(other, "remote", "add", "fork", str(fork))
    _repo_git(other, "fetch", "-q", "fork")
    _repo_git(other, "checkout", "-q", "-b", "main", "fork/main")
    _commit_file(other, "b.txt", "y", "fork advance")
    _repo_git(other, "push", "-q", "fork", "main")

    _repo_git(local, "fetch", "-q", "--all")
    state2 = kb.repo_state_for(kb.get_task(conn, gate))
    assert state2.get("pushed") is False
    assert state2.get("remote_head") is None  # ambiguous -> unknown


def test_repo_state_full_head_and_dirty_covers_untracked_and_staged(conn, tmp_path):
    """repo_state reports the FULL head SHA; dirty covers untracked files and
    staged-but-uncommitted content, not only modified tracked files."""
    import subprocess

    local = _init_repo(tmp_path / "local")
    _commit_file(local)
    full = subprocess.run(
        ["git", "-C", str(local), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert len(full) == 40

    gate = _gate_on_repo(conn, local)
    state = kb.repo_state_for(kb.get_task(conn, gate))
    assert state.get("head") == full
    assert state.get("dirty") is False

    (local / "untracked.txt").write_text("u")  # untracked
    assert kb.repo_state_for(kb.get_task(conn, gate)).get("dirty") is True
    (local / "untracked.txt").unlink()
    (local / "staged.txt").write_text("s")
    _repo_git(local, "add", "staged.txt")  # staged, uncommitted
    assert kb.repo_state_for(kb.get_task(conn, gate)).get("dirty") is True


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


# -- execution witness / fail-closed AUTO (2026-09-04 git-truth delivery) ---
#
# Never persist AUTO/STARTING merely because the resolver classified AUTO or
# an auto_continue event could be written. An AUTO continuation is persisted
# ONLY when the caller supplies a REAL execution witness/materialization
# record (an existing task_runs row from an executor that actually started);
# otherwise the handoff fails closed to APPROVAL_REQUIRED + AWAITING_APPROVAL
# and performs no mutation.


def _executor_witness(conn, mission_task_id, *, action_type="REMEDIATION"):
    """Claim a same-mission executor and persist its action relation.

    Returns ``{"source": "task_run", "run_id": ...}`` referencing the live
    task_runs row the executor's claim created — a real persisted
    materialization record, never an invented one. Both the task link and the
    execution_witness event bind the run to the target mission/action.
    """
    child = kb.create_task(
        conn,
        title="Auto executor",
        assignee="tester",
        parents=[mission_task_id],
    )
    parent = kb.get_task(conn, mission_task_id)
    if parent is not None and parent.status != "done":
        kb.claim_task(conn, mission_task_id)
        kb.complete_task(conn, mission_task_id, result="mission underway")
    claimed = kb.claim_task(conn, child, claimer="test")
    if claimed is None:
        ok, why = kb.promote_task(conn, child, actor="test")
        assert ok, why
        claimed = kb.claim_task(conn, child, claimer="test")
    assert claimed is not None and claimed.current_run_id is not None
    run_id = int(claimed.current_run_id)
    with kb.write_txn(conn):
        kb._append_event(
            conn,
            child,
            kb.EXECUTION_WITNESS_EVENT_KIND,
            {
                "source": "task_run",
                "run_id": run_id,
                "task_id": child,
                "mission_task_id": mission_task_id,
                "actionType": action_type,
            },
            run_id=run_id,
        )
    return {"source": "task_run", "run_id": run_id}


def test_auto_without_execution_witness_fails_closed_to_approval(conn):
    """AUTO-classified next action with NO execution witness must NOT persist
    AUTO/STARTING or an auto_continue event: fail closed to
    APPROVAL_REQUIRED + AWAITING_APPROVAL with explicit evidence."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="QA rejected: changes required.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(conn, gate_task) is True

    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 1
    assert "Decision class: APPROVAL_REQUIRED" in bodies[0]
    assert "Action status: AWAITING_APPROVAL" in bodies[0]
    assert "no execution witness" in bodies[0]

    newest = _newest_handoff_event_payload(conn, umbrella)
    decision = newest["decision"]
    assert decision["actionType"] == "REMEDIATION"  # resolver still says AUTO…
    assert decision["decisionClass"] == kb.APPROVAL_REQUIRED  # …but closed
    assert decision["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert decision["failClosedToApproval"] is True
    assert newest["autoContinue"] is False

    rows = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? LIMIT 1",
        (umbrella, kb.AUTO_CONTINUE_EVENT_KIND),
    ).fetchone()
    assert rows is None  # no fabricated continuation


def test_auto_with_execution_witness_persists_starting_and_continue(conn):
    """AUTO + real persisted execution witness (an executor task run) ->
    STARTING is legitimate: decision AUTO + actionStatus STARTING +
    auto_continue event referencing the run; replay stays idempotent."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="QA rejected: changes required.",
    )
    witness = _executor_witness(conn, umbrella)
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(
        conn, gate_task, execution_witness=witness,
    ) is True

    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 1
    assert "Decision class: AUTO" in bodies[0]
    assert "Action status: STARTING" in bodies[0]

    newest = _newest_handoff_event_payload(conn, umbrella)
    assert newest["decision"]["decisionClass"] == kb.AUTO
    assert newest["decision"]["actionStatus"] == kb.ACTION_STATUS_STARTING
    assert newest["autoContinue"] is True
    assert newest["execution"]["run_id"] == witness["run_id"]

    continue_events = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
        (umbrella, kb.AUTO_CONTINUE_EVENT_KIND),
    ).fetchall()
    assert len(continue_events) == 1
    payload = json.loads(continue_events[0]["payload"])
    assert payload["execution"]["run_id"] == witness["run_id"]

    # Replay on a fresh connection never duplicates the witnessed launch.
    with kb.connect() as conn2:
        assert kb.emit_terminal_handoffs_if_due(conn2) == []
        assert len(_handoff_comment_bodies(conn2, umbrella)) == 1


def test_cross_mission_execution_witness_fails_closed_without_auto_continue(conn):
    """A live run related to another mission cannot authorize this mission's
    AUTO/STARTING transition or be described as same-mission recovery."""
    target_mission, target_gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="QA rejected: changes required.",
    )
    other_mission = kb.create_task(conn, title="Other mission", role="umbrella")
    witness = _executor_witness(conn, other_mission)

    gate_task = kb.get_task(conn, target_gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(
        conn, gate_task, execution_witness=witness,
    ) is True

    newest = _newest_handoff_event_payload(conn, target_mission)
    assert newest["decision"]["decisionClass"] == kb.APPROVAL_REQUIRED
    assert newest["decision"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert newest["decision"]["failClosedToApproval"] is True
    assert newest["discussion"]["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert newest["discussion"]["reason"] != "auto_continuation_active"
    assert newest["execution"] is None
    assert newest["autoContinue"] is False
    assert "de la même mission" not in _handoff_comment_bodies(conn, target_mission)[0]
    assert conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? LIMIT 1",
        (target_mission, kb.AUTO_CONTINUE_EVENT_KIND),
    ).fetchone() is None


def test_accepted_dirty_feature_branch_repo_policy_requires_approval(conn, tmp_path):
    """Repository policy wins over autonomy: a dirty accepted gate in a repo
    that forbids commit/push without owner approval persists DELIVERY_CHECKPOINT
    APPROVAL_REQUIRED + AWAITING_APPROVAL (feature branch!), performs no Git
    mutation, and carries the French delivery-waiting approval block."""
    repo = _init_repo(tmp_path / "repo", branch="feat/x")
    _commit_file(repo)
    (repo / "candidate.txt").write_text("accepted work")  # dirty

    umbrella = kb.create_task(conn, title="Mission", role="umbrella")
    gate = kb.create_task(
        conn, title="Gate", role="gate", parents=[umbrella],
        workspace_kind="dir", workspace_path=str(repo),
    )
    kb.claim_task(conn, umbrella)
    kb.complete_task(conn, umbrella, result="underway")
    assert kb.claim_task(conn, gate) is not None
    assert kb.complete_task(conn, gate, result="ACCEPTED", summary="All green.")
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None

    policy = kb.autonomy_policy_from_config({
        "kanban": {"autonomy": {
            "approvals": {"commit": "required", "push": "required"},
        }}
    })
    assert kb.emit_terminal_handoff(
        conn, gate_task, autonomy_policy=policy,
    ) is True

    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 1
    body = bodies[0]
    assert "Decision class: APPROVAL_REQUIRED" in body
    assert "Action status: AWAITING_APPROVAL" in body
    assert "LIVRAISON EN ATTENTE" in body
    assert "uncommitted" in body
    assert "no remote update" in body
    assert "authorization" in body
    assert "RECOMMENDED OWNER DECISION: APPROVE DELIVERY CHECKPOINT" in body

    newest = _newest_handoff_event_payload(conn, umbrella)
    assert newest["next_action"]["type"] == "DELIVERY_CHECKPOINT"
    assert newest["decision"]["decisionClass"] == kb.APPROVAL_REQUIRED
    assert newest["decision"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    rows = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? LIMIT 1",
        (umbrella, kb.AUTO_CONTINUE_EVENT_KIND),
    ).fetchone()
    assert rows is None

    # No Git mutation: still dirty, still exactly the same HEAD, no new files
    # staged/committed by the handoff machinery.
    import subprocess

    head_now = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head_now == newest["repo_state"]["head"]
    assert kb.repo_state_for(kb.get_task(conn, gate))["dirty"] is True


# -- STARTING handoff watchdog (2026-09-04 git-truth delivery) --------------
#
# A persisted STARTING auto-continuation that never gained a live execution
# run/witness/heartbeat is a stuck claim. The watchdog (called from the
# existing dispatch tick) corrects it to APPROVAL_REQUIRED + AWAITING_APPROVAL
# with explicit corrective evidence — bounded (only STARTING handoffs) and
# idempotent (one correction per handoff event).


def _plant_legacy_starting_handoff(conn, umbrella, gate, *, run_id=None):
    """Plant a STARTING AUTO handoff event directly (as pre-fix code wrote):
    resolver AUTO with no execution witness materialization."""
    with kb.write_txn(conn):
        kb._append_event(
            conn, umbrella, kb.HANDOFF_EVENT_KIND,
            {
                "marker": kb.HANDOFF_MARKER,
                "gate_id": gate,
                "verdict": "REJECTED / CHANGES REQUIRED",
                "repo_state": {},
                "next_action": {"type": "REMEDIATION", "requiresApproval": False},
                "decision": {
                    "actionType": "REMEDIATION",
                    "decisionClass": kb.AUTO,
                    "requiresApproval": False,
                    "actionStatus": kb.ACTION_STATUS_STARTING,
                    "analysis": "Final gate completed.",
                },
                "execution": (
                    {"source": "task_run", "run_id": run_id} if run_id else None
                ),
                "autoContinue": True,
            },
        )


def _watchdog_events(conn, task_id):
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id",
        (task_id, kb.TERMINAL_WATCHDOG_EVENT_KIND),
    ).fetchall()


def test_watchdog_corrects_starting_without_witness_and_is_idempotent(conn):
    """A STARTING handoff with NO execution witness is corrected to approval
    evidence; the watchdog never corrects the same handoff twice."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="QA rejected: changes required.",
    )
    conn.execute(
        "UPDATE tasks SET assignee = 'hotelos-lead' WHERE id = ?", (umbrella,)
    )
    _plant_legacy_starting_handoff(conn, umbrella, gate)
    planted_at = conn.execute(
        "SELECT created_at FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1", (umbrella, kb.HANDOFF_EVENT_KIND)
    ).fetchone()[0]

    corrected = kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(planted_at) + 120
    )
    assert corrected == [umbrella]

    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 1
    assert "WATCHDOG" in bodies[0]
    assert "AWAITING_APPROVAL" in bodies[0]
    headings = [
        "## NEXT ACTION",
        "## DISCUSSION STATUS",
        "## DISCUSSION ACTION",
        "## DECISION CLASS",
        "## ACTION STATUS",
    ]
    assert [bodies[0].index(h) for h in headings] == sorted(
        bodies[0].index(h) for h in headings
    )

    events = _watchdog_events(conn, umbrella)
    assert len(events) == 1
    payload = json.loads(events[0]["payload"])
    assert payload["from"]["actionStatus"] == kb.ACTION_STATUS_STARTING
    assert payload["to"]["decisionClass"] == kb.APPROVAL_REQUIRED
    assert payload["to"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert payload["discussion"]["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert "Action du propriétaire requise" in payload["discussion"]["action"]

    # Idempotent: the same STARTING handoff is never corrected twice.
    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(planted_at) + 240
    ) == []
    assert len(_watchdog_events(conn, umbrella)) == 1


def test_watchdog_skips_starting_with_live_execution_witness(conn):
    """STARTING with a live executor run (running, claim unexpired) is real
    materialization — the watchdog must not touch it."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="QA rejected: changes required.",
    )
    witness = _executor_witness(conn, umbrella)
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(
        conn, gate_task, execution_witness=witness,
    ) is True

    assert kb.watchdog_terminal_handoffs(conn) == []
    # Even after the full watchdog window, the live executor run is real
    # materialization: STARTING stays current and is never corrected.
    import time

    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(time.time()) + 120
    ) == []
    assert _watchdog_events(conn, umbrella) == []
    assert len(_handoff_comment_bodies(conn, umbrella)) == 1


def test_watchdog_corrects_starting_whose_executor_run_died(conn):
    """STARTING whose witness run crashed/ended without completion is no
    longer live -> corrected to approval evidence."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="QA rejected: changes required.",
    )
    witness = _executor_witness(conn, umbrella)
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(
        conn, gate_task, execution_witness=witness,
    ) is True

    # The executor run dies (crash) without completing the auto action.
    import time

    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET status = 'crashed', outcome = 'crashed', "
            "ended_at = ? WHERE id = ?",
            (int(time.time()), witness["run_id"]),
        )

    corrected = kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(time.time()) + 120
    )
    assert corrected == [umbrella]
    events = _watchdog_events(conn, umbrella)
    assert len(events) == 1
    payload = json.loads(events[0]["payload"])
    assert payload["reason"]
    assert payload["to"]["decisionClass"] == kb.APPROVAL_REQUIRED
    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(time.time()) + 240
    ) == []


# -- reconciliation: derived Git fields / structured truth (2026-09-04) -----
#
# Recompute refreshes the persisted repo_state Git fields AND the resolved
# next-action whenever either moved — historical prose in a comment/result can
# never override the structured verdict or current Git truth.


def test_recompute_refreshes_git_fields_when_head_advances_still_dirty(conn, tmp_path):
    """Repo advances (new commit) while remaining dirty: verdict and next-action
    type are unchanged (DELIVERY_CHECKPOINT) but the persisted repo_state Git
    fields are stale — recompute must refresh them (RECOMPUTED), once."""
    import subprocess

    repo = _init_repo(tmp_path / "repo", branch="main")
    _commit_file(repo)
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

    assert kb.emit_terminal_handoff(conn, gate_task) is True
    first = _newest_handoff_event_payload(conn, umbrella)
    assert _next_action_type(first) == "DELIVERY_CHECKPOINT"
    first_head = first["repo_state"]["head"]

    # Repo advances: b.txt is committed, c.txt stays dirty -> NEW head, still
    # dirty, next-action type unchanged.
    _commit_file(repo, "b.txt", "y", "advance")
    (repo / "c.txt").write_text("z")
    head_now = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head_now != first_head

    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is True
    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 2
    assert "RECOMPUTED" in bodies[1]
    newest = _newest_handoff_event_payload(conn, umbrella)
    assert _next_action_type(newest) == "DELIVERY_CHECKPOINT"
    assert newest["repo_state"]["head"] == head_now
    assert newest["repo_state"]["dirty"] is True

    # Idempotent: persisted fields now match derived Git truth.
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is False
    assert len(_handoff_comment_bodies(conn, umbrella)) == 2


def test_recompute_historical_prose_cannot_override_structured_git_truth(conn, tmp_path):
    """A handoff whose stored repo_state claims pushed=True (historical/incorrect
    prose record) must be corrected by recompute against real Git truth
    (committed but NOT pushed) — structured verdict + current Git win."""
    import subprocess

    repo = _init_repo(tmp_path / "repo", branch="main")
    _commit_file(repo)  # committed, NO remote -> pushed False

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

    # Plant a stale handoff event whose prose-era repo_state wrongly claims
    # the branch is pushed (short head, no remote fields).
    with kb.write_txn(conn):
        kb._append_event(
            conn, umbrella, kb.HANDOFF_EVENT_KIND,
            {
                "marker": kb.HANDOFF_MARKER,
                "gate_id": gate,
                "verdict": "ACCEPTED",
                "repo_state": {
                    "branch": "main", "head": "deadbee",
                    "dirty": False, "committed": True, "pushed": True,
                    "unpushed_commits": 0,
                },
                "next_action": {"type": "PUSH_CHECKPOINT", "requiresApproval": True},
                "decision": {
                    "actionType": "PUSH_CHECKPOINT",
                    "decisionClass": kb.APPROVAL_REQUIRED,
                    "actionStatus": kb.ACTION_STATUS_AWAITING_APPROVAL,
                },
                "autoContinue": False,
            },
        )
    with kb.write_txn(conn):
        kb.add_comment(
            conn, umbrella, kb.HANDOFF_AUTHOR,
            f"{kb.HANDOFF_MARKER}\nDelivered and pushed to origin already.",
        )

    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is True
    newest = _newest_handoff_event_payload(conn, umbrella)
    assert newest["verdict"] == "ACCEPTED"
    assert newest["next_action"]["type"] == "PUSH_CHECKPOINT"
    assert newest["repo_state"]["pushed"] is False  # structured Git truth
    assert newest["repo_state"]["head"] != "deadbee"
    full = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert newest["repo_state"]["head"] == full
    bodies = _handoff_comment_bodies(conn, umbrella)
    assert len(bodies) == 2
    assert "RECOMPUTED" in bodies[1]

    # Stored fields now match Git truth -> no further corrective emissions.
    assert kb.emit_terminal_handoff(conn, gate_task, recompute=True) is False
    assert len(_handoff_comment_bodies(conn, umbrella)) == 2


# -- discussion lifecycle projection (A-H) ---------------------------------


def test_discussion_a_completed_without_follow_up_is_archive_ready():
    discussion = kb.resolve_discussion_lifecycle(
        {
            "gate_id": "t_gate",
            "gate_status": "done",
            "active_workers": [],
            "pending_work": [],
            "blockers": [],
        },
        next_action={"type": "NONE", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )

    assert discussion["status"] == kb.DISCUSSION_ARCHIVE_READY
    assert discussion["action"] == (
        "Aucune action requise : mission terminée, aucun suivi actif détecté. "
        "Archivage autorisé."
    )


def test_discussion_b_separate_future_increment_does_not_keep_mission_open(conn):
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    future = kb.create_task(conn, title="Independent future increment")
    assert future not in {umbrella, gate}

    snapshot = kb.synthesize_terminal_handoff(
        conn,
        kb.get_task(conn, gate),
        kb.get_task(conn, umbrella),
    )
    discussion = kb.resolve_discussion_lifecycle(
        snapshot,
        next_action={"type": "NONE", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )

    assert snapshot["pending_work"] == []
    assert discussion["status"] == kb.DISCUSSION_ARCHIVE_READY


def test_discussion_c_nested_running_worker_keeps_mission_open(conn):
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    phase = kb.create_task(conn, title="Mission phase", parents=[umbrella])
    assert kb.claim_task(conn, phase) is not None
    assert kb.complete_task(conn, phase, result="phase complete")
    worker = kb.create_task(
        conn,
        title="Nested active worker",
        assignee="hotelos-lead",
        parents=[phase],
    )
    assert kb.claim_task(conn, worker) is not None

    snapshot = kb.synthesize_terminal_handoff(
        conn,
        kb.get_task(conn, gate),
        kb.get_task(conn, umbrella),
    )
    discussion = kb.resolve_discussion_lifecycle(
        snapshot,
        next_action={"type": "AWAITING_WORKERS", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_STARTING,
    )

    assert any(item.startswith(worker) for item in snapshot["active_workers"])
    assert discussion["status"] == kb.DISCUSSION_KEEP_OPEN
    assert "ne pas archiver" in discussion["action"]


def test_discussion_d_auto_recovery_with_live_witness_keeps_open():
    discussion = kb.resolve_discussion_lifecycle(
        {
            "gate_id": "t_gate",
            "gate_status": "done",
            "active_workers": [],
            "pending_work": [],
            "blockers": [],
            "state_conflicts": [],
        },
        next_action={"type": "REMEDIATION", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_STARTING,
        execution_witness={
            "source": "task_run",
            "run_id": 42,
            "task_id": "t_recovery",
            "status": "running",
        },
        execution_is_live=True,
    )

    assert discussion["status"] == kb.DISCUSSION_KEEP_OPEN
    assert discussion["reason"] == "auto_continuation_active"
    assert "recovery AUTO" in discussion["action"]


def test_discussion_d_auto_recovery_without_witness_fails_closed_to_owner():
    discussion = kb.resolve_discussion_lifecycle(
        {
            "gate_id": "t_gate",
            "gate_status": "done",
            "active_workers": [],
            "pending_work": [],
            "blockers": [],
            "state_conflicts": [],
        },
        next_action={"type": "REMEDIATION", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_STARTING,
    )

    assert discussion["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert discussion["reason"] == "auto_start_without_live_execution_witness"
    assert "État non démontré" in discussion["action"]


def test_discussion_e_delivery_approval_requires_owner_action():
    discussion = kb.resolve_discussion_lifecycle(
        {
            "gate_id": "t_gate",
            "gate_status": "done",
            "active_workers": [],
            "pending_work": [],
            "blockers": [],
            "state_conflicts": [],
        },
        next_action={"type": "DELIVERY_CHECKPOINT", "requiresApproval": True},
        decision_class=kb.APPROVAL_REQUIRED,
        action_status=kb.ACTION_STATUS_AWAITING_APPROVAL,
    )

    assert discussion["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert discussion["reason"] == "owner_approval_required"
    assert "livraison" in discussion["action"]
    assert "Aucune modification" in discussion["action"]


def test_discussion_f_merge_approval_beats_auto_mode():
    discussion = kb.resolve_discussion_lifecycle(
        {
            "gate_id": "t_gate",
            "gate_status": "done",
            "active_workers": ["t_worker (hotelos-lead)"],
            "pending_work": [],
            "blockers": [],
            "state_conflicts": [],
        },
        next_action={"type": "INTEGRATION_REVIEW", "requiresApproval": True},
        decision_class=kb.APPROVAL_REQUIRED,
        action_status=kb.ACTION_STATUS_AWAITING_APPROVAL,
    )

    assert discussion["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert discussion["reason"] == "owner_approval_required"
    assert "merge" in discussion["action"]
    assert "Aucun merge" in discussion["action"]


def test_discussion_g_same_mission_remediation_pending_keeps_open(conn):
    umbrella, gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="QA rejected: remediation required.",
    )
    remediation = kb.create_task(
        conn,
        title="Corrective remediation",
        assignee="hotelos-lead",
        parents=[umbrella],
    )

    snapshot = kb.synthesize_terminal_handoff(
        conn,
        kb.get_task(conn, gate),
        kb.get_task(conn, umbrella),
    )
    discussion = kb.resolve_discussion_lifecycle(
        snapshot,
        next_action={"type": "REMEDIATION", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )

    assert any(item.startswith(remediation) for item in snapshot["pending_work"])
    assert discussion["status"] == kb.DISCUSSION_KEEP_OPEN
    assert discussion["reason"] == "same_mission_work_pending"
    assert "remediation" in discussion["action"]


def test_discussion_h_historical_terminal_prose_does_not_reopen_mission():
    snapshot = {
        "mission_task_id": "t_mission",
        "gate_id": "t_gate",
        "gate_status": "archived",
        "active_workers": [],
        "pending_work": [],
        "blockers": [],
        "state_conflicts": [],
        "historical_comments": [
            "A worker was running and remediation remained open last month."
        ],
    }
    first = kb.resolve_discussion_lifecycle(
        snapshot,
        next_action={"type": "NONE", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )
    replay = kb.resolve_discussion_lifecycle(
        snapshot,
        next_action={"type": "NONE", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )

    assert first == replay
    assert first["status"] == kb.DISCUSSION_ARCHIVE_READY
    assert first["evidence"]["missionTaskId"] == "t_mission"
    assert first["evidence"]["gateId"] == "t_gate"


def test_terminal_handoff_renders_and_persists_additive_discussion_fields(conn):
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    gate_task = kb.get_task(conn, gate)
    umbrella_task = kb.get_task(conn, umbrella)
    assert gate_task is not None and umbrella_task is not None
    snapshot = kb.synthesize_terminal_handoff(conn, gate_task, umbrella_task)
    existing_decision = kb.resolve_next_action(snapshot)

    assert kb.emit_terminal_handoff(conn, gate_task) is True

    body = _handoff_comment_bodies(conn, umbrella)[0]
    headings = [
        "## NEXT ACTION",
        "## DISCUSSION STATUS",
        "## DISCUSSION ACTION",
        "## DECISION CLASS",
        "## ACTION STATUS",
    ]
    positions = [body.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Action du propriétaire requise" in body

    payload = _newest_handoff_event_payload(conn, umbrella)
    assert payload["next_action"]["type"] == existing_decision["type"]
    assert payload["decision"]["decisionClass"] == existing_decision["decisionClass"]
    assert payload["decision"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert payload["discussion"]["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert payload["discussion"]["action"]
    assert payload["discussion"]["evidence"]["missionTaskId"] == umbrella

    with kb.connect() as conn2:
        assert kb.emit_terminal_handoffs_if_due(conn2) == []
        assert len(_handoff_comment_bodies(conn2, umbrella)) == 1


def test_discussion_conflicting_run_identity_fails_closed_to_owner():
    discussion = kb.resolve_discussion_lifecycle(
        {
            "mission_task_id": "t_mission",
            "gate_id": "t_gate",
            "gate_status": "done",
            "active_workers": [],
            "pending_work": [],
            "blockers": [],
            "state_conflicts": [
                "t_worker (running task without coherent live run)"
            ],
        },
        next_action={"type": "NONE", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )

    assert discussion["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert discussion["reason"] == "insufficient_or_conflicting_evidence"
    assert "vérification requise" in discussion["action"]


def test_discussion_terminal_task_with_live_run_fails_closed(conn):
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    worker = kb.create_task(
        conn,
        title="Contradictory worker state",
        assignee="hotelos-lead",
        parents=[umbrella],
    )
    assert kb.claim_task(conn, worker) is not None
    conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (worker,))

    gate_task = kb.get_task(conn, gate)
    umbrella_task = kb.get_task(conn, umbrella)
    assert gate_task is not None and umbrella_task is not None
    snapshot = kb.synthesize_terminal_handoff(conn, gate_task, umbrella_task)
    discussion = kb.resolve_discussion_lifecycle(
        snapshot,
        next_action={"type": "NONE", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )

    assert any(item.startswith(worker) for item in snapshot["state_conflicts"])
    assert discussion["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert discussion["reason"] == "insufficient_or_conflicting_evidence"


@pytest.mark.parametrize(
    "run_anomaly",
    ["terminal_orphan", "terminal_non_current", "multiple_live"],
)
def test_discussion_all_descendant_live_run_anomalies_fail_closed(
    conn, run_anomaly,
):
    """Every descendant run participates in reconciliation: orphan,
    non-current, and multiple live runs are contradictory and can never
    produce a false ARCHIVE_READY result."""
    import time

    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    worker = kb.create_task(
        conn,
        title="Worker with run anomaly",
        assignee="hotelos-lead",
        parents=[umbrella],
    )
    claimed = kb.claim_task(conn, worker)
    assert claimed is not None and claimed.current_run_id is not None
    current_run_id = int(claimed.current_run_id)
    now = int(time.time())

    if run_anomaly == "terminal_orphan":
        conn.execute(
            "UPDATE tasks SET status = 'done', current_run_id = NULL WHERE id = ?",
            (worker,),
        )
    elif run_anomaly == "terminal_non_current":
        conn.execute(
            "UPDATE task_runs SET status = 'done', outcome = 'completed', "
            "ended_at = ? WHERE id = ?",
            (now, current_run_id),
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (worker,))
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_expires, "
            "started_at) VALUES (?, ?, 'running', ?, ?)",
            (worker, "hotelos-lead", now + 300, now),
        )
    else:
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_expires, "
            "started_at) VALUES (?, ?, 'running', ?, ?)",
            (worker, "hotelos-lead", now + 300, now),
        )

    gate_task = kb.get_task(conn, gate)
    umbrella_task = kb.get_task(conn, umbrella)
    assert gate_task is not None and umbrella_task is not None
    snapshot = kb.synthesize_terminal_handoff(conn, gate_task, umbrella_task)
    discussion = kb.resolve_discussion_lifecycle(
        snapshot,
        next_action={"type": "NONE", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )

    assert any(worker in conflict for conflict in snapshot["state_conflicts"])
    assert discussion["status"] == kb.DISCUSSION_OWNER_ACTION_REQUIRED
    assert discussion["status"] != kb.DISCUSSION_ARCHIVE_READY


def test_discussion_no_false_archive_when_descendant_runs_cleanly_ended(conn):
    """Terminal descendant with a coherent ended run is NOT a state conflict:
    cleanly completed runs never block ARCHIVE_READY (no false archive)."""
    import time

    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    worker = kb.create_task(
        conn,
        title="Completed worker",
        assignee="hotelos-lead",
        parents=[umbrella],
    )
    claimed = kb.claim_task(conn, worker)
    assert claimed is not None and claimed.current_run_id is not None
    run_id = int(claimed.current_run_id)
    assert kb.complete_task(conn, worker, result="done") is True
    conn.execute(
        "UPDATE task_runs SET status = 'done', outcome = 'completed', "
        "ended_at = ? WHERE id = ?",
        (int(time.time()), run_id),
    )

    gate_task = kb.get_task(conn, gate)
    umbrella_task = kb.get_task(conn, umbrella)
    assert gate_task is not None and umbrella_task is not None
    snapshot = kb.synthesize_terminal_handoff(conn, gate_task, umbrella_task)
    discussion = kb.resolve_discussion_lifecycle(
        snapshot,
        next_action={"type": "NONE", "requiresApproval": False},
        decision_class=kb.AUTO,
        action_status=kb.ACTION_STATUS_EXECUTED,
    )

    assert snapshot["state_conflicts"] == []
    assert snapshot["active_workers"] == []
    assert snapshot["pending_work"] == []
    assert discussion["status"] == kb.DISCUSSION_ARCHIVE_READY


# -- terminal probe/test lifecycle cleanup ---------------------------------


def test_mission_kind_is_structured_and_validated_at_creation(conn):
    probe = kb.create_task(
        conn, title="Synthetic lifecycle fixture", role="umbrella", mission_kind="probe"
    )
    test = kb.create_task(
        conn, title="Synthetic test fixture", role="umbrella", mission_kind="TEST"
    )

    assert kb.get_task(conn, probe).mission_kind == "probe"
    assert kb.get_task(conn, test).mission_kind == "test"
    with pytest.raises(ValueError, match="mission_kind"):
        kb.create_task(
            conn, title="Not a probe", role="umbrella", mission_kind="synthetic"
        )


def test_watchdog_waits_full_window_then_reemits_corrected_handoff(conn):
    umbrella, gate = _run_mission(
        conn,
        verdict_result="REJECT",
        verdict_summary="QA rejected: changes required.",
    )
    conn.execute(
        "UPDATE tasks SET assignee = 'hotelos-lead' WHERE id = ?", (umbrella,)
    )
    _plant_legacy_starting_handoff(conn, umbrella, gate)
    planted = conn.execute(
        "SELECT id, created_at FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (umbrella, kb.HANDOFF_EVENT_KIND),
    ).fetchone()

    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(planted["created_at"]) + 119
    ) == []
    assert _newest_handoff_event_payload(conn, umbrella)["decision"][
        "actionStatus"
    ] == kb.ACTION_STATUS_STARTING

    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(planted["created_at"]) + 120
    ) == [umbrella]
    corrected = _newest_handoff_event_payload(conn, umbrella)
    assert corrected["recomputed"] is True
    assert corrected["recomputed_from_handoff_event_id"] == int(planted["id"])
    assert corrected["decision"]["decisionClass"] == kb.APPROVAL_REQUIRED
    assert corrected["decision"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert kb.get_task(conn, umbrella).status == "done"
    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(planted["created_at"]) + 240
    ) == []


def test_terminal_probe_auto_archive_preserves_history_and_is_idempotent(conn):
    umbrella = kb.create_task(
        conn, title="Synthetic fixture", role="umbrella", mission_kind="probe"
    )
    gate = kb.create_task(
        conn, title="Fixture gate", role="gate", parents=[umbrella]
    )
    generic = kb.create_task(
        conn, title="Corrective remediation", parents=[umbrella]
    )
    assert kb.claim_task(conn, umbrella) is not None
    assert kb.complete_task(conn, umbrella, result="fixture complete")
    assert kb.claim_task(conn, gate) is not None
    assert kb.complete_task(conn, gate, result="ACCEPTED", summary="All checks green.")
    assert kb.claim_task(conn, generic) is not None
    assert kb.complete_task(conn, generic, result="done")
    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]

    before = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("task_runs", "task_comments", "task_links")
    }
    # SEC-PROBE-002: the bounded window now applies whatever the handoff
    # actionStatus, so the just-emitted handoff must be past the window first.
    planted_at = _newest_handoff_created_at(conn, umbrella)
    assert kb.cleanup_terminal_probe_missions(
        conn, window_seconds=120, now=planted_at + 120
    ) == [umbrella]
    assert kb.get_task(conn, umbrella).status == "archived"
    assert kb.get_task(conn, gate).status == "archived"
    assert kb.get_task(conn, generic).status == "done"
    assert conn.execute(
        "SELECT id FROM tasks WHERE role = 'umbrella' AND status != 'archived' "
        "AND id = ?", (umbrella,)
    ).fetchone() is None
    after = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("task_runs", "task_comments", "task_links")
    }
    assert after == before
    archived_events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind = 'archived' "
        "AND task_id IN (?, ?)", (umbrella, gate)
    ).fetchone()[0]
    assert archived_events == 2

    assert kb.cleanup_terminal_probe_missions(
        conn, window_seconds=120, now=planted_at + 240
    ) == []
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind = 'archived' "
        "AND task_id IN (?, ?)", (umbrella, gate)
    ).fetchone()[0] == archived_events


def test_stale_starting_probe_waits_then_archives_in_watchdog_tick(conn):
    umbrella = kb.create_task(
        conn, title="Synthetic rejected fixture", role="umbrella", mission_kind="test"
    )
    gate = kb.create_task(
        conn, title="Fixture gate", role="gate", parents=[umbrella]
    )
    assert kb.claim_task(conn, umbrella) is not None
    assert kb.complete_task(conn, umbrella, result="fixture complete")
    assert kb.claim_task(conn, gate) is not None
    assert kb.complete_task(
        conn, gate, result="REJECT", summary="QA rejected: changes required."
    )
    _plant_legacy_starting_handoff(conn, umbrella, gate)
    planted_at = conn.execute(
        "SELECT created_at FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1", (umbrella, kb.HANDOFF_EVENT_KIND)
    ).fetchone()[0]

    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(planted_at) + 119
    ) == []
    assert kb.get_task(conn, umbrella).status == "done"
    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(planted_at) + 120
    ) == [umbrella]
    assert kb.get_task(conn, umbrella).status == "archived"
    assert kb.get_task(conn, gate).status == "archived"
    lifecycle = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
        (umbrella, kb.TERMINAL_PROBE_CLEANUP_EVENT_KIND),
    ).fetchall()
    assert len(lifecycle) == 1
    assert json.loads(lifecycle[0]["payload"])["classification"] == (
        kb.DISCUSSION_ARCHIVE_READY
    )
    assert _watchdog_events(conn, umbrella) == []
    assert kb.watchdog_terminal_handoffs(
        conn, window_seconds=120, now=int(planted_at) + 240
    ) == []


def test_cleanup_never_archives_unproven_legacy_or_real_or_owner_attended_missions(conn):
    """SEC-PROBE-001 fail-closed: an umbrella mission whose ONLY probe hint is
    the legacy_derived identity (assignee NULL) is NOT auto-archived when it
    lacks the stale-STARTING signature, a structured mission_kind, or an
    ARCHIVE_READY journal — it is indistinguishable from a real mission whose
    umbrella simply has no assignee yet. Real (assigned) and owner-attended
    missions stay done too."""
    # Unproven legacy look-alike: done umbrella, done ACCEPTED gate, NO
    # assignee, NO stale-STARTING handoff, NO mission_kind. Pre-fix this was
    # auto-archived immediately (the fail-open Security reproduced).
    legacy, legacy_gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )

    real, real_gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", ("hotelos-lead", real))

    attended = kb.create_task(
        conn, title="Attended fixture", role="umbrella", mission_kind="probe"
    )
    attended_gate = kb.create_task(
        conn, title="Attended gate", role="gate", parents=[attended]
    )
    assert kb.claim_task(conn, attended) is not None
    assert kb.complete_task(conn, attended, result="fixture complete")
    assert kb.claim_task(conn, attended_gate) is not None
    assert kb.complete_task(
        conn, attended_gate, result="ACCEPTED", summary="All checks green."
    )
    terminal_at = max(
        kb.get_task(conn, attended).completed_at,
        kb.get_task(conn, attended_gate).completed_at,
    )
    comment_id = kb.add_comment(
        conn, attended, "user", "Keep this synthetic mission for manual review."
    )
    conn.execute(
        "UPDATE task_comments SET created_at = ? WHERE id = ?",
        (int(terminal_at) + 1, comment_id),
    )

    # All three remain done no matter how far past terminality we scan.
    assert kb.cleanup_terminal_probe_missions(conn, now=int(terminal_at) + 10_000) == []
    for tid in (legacy, legacy_gate, real, real_gate, attended, attended_gate):
        row = kb.get_task(conn, tid)
        assert row is not None and row.status == "done"
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind = 'archived' "
        "AND task_id IN (?, ?, ?, ?, ?, ?)",
        (legacy, legacy_gate, real, real_gate, attended, attended_gate),
    ).fetchone()[0] == 0


def test_cleanup_never_archives_mission_with_active_live_descendant(conn):
    umbrella = kb.create_task(
        conn, title="Synthetic fixture", role="umbrella", mission_kind="test"
    )
    gate = kb.create_task(
        conn, title="Fixture gate", role="gate", parents=[umbrella]
    )
    live = kb.create_task(
        conn, title="Remediation worker", assignee="hotelos-lead", parents=[umbrella]
    )
    assert kb.claim_task(conn, umbrella) is not None
    assert kb.complete_task(conn, umbrella, result="fixture complete")
    assert kb.claim_task(conn, gate) is not None
    assert kb.complete_task(conn, gate, result="ACCEPTED", summary="All checks green.")
    claimed = kb.claim_task(conn, live)
    assert claimed is not None and claimed.status == "running"
    assert kb.get_task(conn, umbrella).status == "done"

    assert kb.cleanup_terminal_probe_missions(conn) == []
    assert kb.get_task(conn, umbrella).status == "done"
    assert kb.get_task(conn, gate).status == "done"
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind = 'archived'"
    ).fetchone()[0] == 0


# -- SEC-PROBE-001/002 remediation: fail-closed legacy probe identity --------
#
# Security review t_93555643 (#246): the legacy_derived identity
# ("umbrella.assignee IS NULL") plus a window that only applied to STARTING
# handoffs let a REAL unassigned umbrella mission whose gate was ACCEPTED and
# whose terminal handoff is AWAITING_APPROVAL / APPROVAL_REQUIRED be
# auto-archived immediately, with no bounded window and no owner action.
# Remediation direction (Product fail-closed): legacy_derived identity alone
# never authorizes auto-archive. The ONLY authorized legacy paths are
#  (a) the stale-STARTING signature (newest handoff STARTING + window elapsed),
#  (b) a structured mission_kind marker, or
#  (c) an already-journaled ARCHIVE_READY (idempotent resume).
# A card whose newest terminal_handoff is APPROVAL_REQUIRED / AWAITING_APPROVAL
# / OWNER_ACTION_REQUIRED is NEVER archived without a structured mission_kind.
# The bounded window now applies regardless of the handoff actionStatus.


def _plant_owner_approval_handoff(conn, umbrella, gate, *, created_at=None):
    """Plant the fail-closed terminal handoff a REAL mission receives after
    its final gate: APPROVAL_REQUIRED / AWAITING_APPROVAL, owner action
    required, no auto-continuation (mirrors the watchdog-corrected payload)."""
    with kb.write_txn(conn):
        kb._append_event(
            conn, umbrella, kb.HANDOFF_EVENT_KIND,
            {
                "marker": kb.HANDOFF_MARKER,
                "gate_id": gate,
                "verdict": "ACCEPTED",
                "repo_state": {},
                "next_action": {"type": "DELIVERY_CHECKPOINT", "requiresApproval": True},
                "decision": {
                    "actionType": "DELIVERY_CHECKPOINT",
                    "decisionClass": kb.APPROVAL_REQUIRED,
                    "requiresApproval": True,
                    "actionStatus": kb.ACTION_STATUS_AWAITING_APPROVAL,
                    "ownerAction": "REQUIRED",
                    "failClosedToApproval": True,
                },
                "autoContinue": False,
                "ownerAction": "REQUIRED",
                "actionStatus": kb.ACTION_STATUS_AWAITING_APPROVAL,
            },
        )
        if created_at is not None:
            conn.execute(
                "UPDATE task_events SET created_at = ? WHERE id = ("
                "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = ?"
                ")",
                (int(created_at), umbrella, kb.HANDOFF_EVENT_KIND),
            )


def _newest_handoff_created_at(conn, task_id) -> int:
    row = conn.execute(
        "SELECT created_at FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, kb.HANDOFF_EVENT_KIND),
    ).fetchone()
    assert row is not None
    return int(row["created_at"])


def test_cleanup_never_archives_real_unassigned_umbrella_with_owner_approval(conn):
    """SEC-PROBE-001 regression (scenario H reinforced): a REAL umbrella
    mission with NO assignee whose gate was ACCEPTED and whose terminal handoff
    is AWAITING_APPROVAL (owner action required) must NEVER be auto-archived —
    not immediately, not after the window, without a structured mission_kind.
    """
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    # This is a REAL mission: no assignee set (Security reproduction), gate
    # done ACCEPTED, and the dispatcher emitted the fail-closed handoff.
    _plant_owner_approval_handoff(conn, umbrella, gate)
    u = kb.get_task(conn, umbrella)
    g = kb.get_task(conn, gate)
    assert u is not None and g is not None and u.completed_at and g.completed_at
    terminal_at = max(int(u.completed_at), int(g.completed_at))

    # Even long after terminality AND after the bounded window, legacy-derived
    # identity alone must not authorize archiving an owner-approval card.
    assert kb.cleanup_terminal_probe_missions(
        conn, now=int(terminal_at) + 10_000
    ) == []
    assert kb.get_task(conn, umbrella).status == "done"
    assert kb.get_task(conn, gate).status == "done"
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind = 'archived'"
    ).fetchone()[0] == 0


def test_cleanup_never_archives_real_unassigned_umbrella_with_approval_required(conn):
    """SEC-PROBE-001: the same protection holds when the newest handoff is
    APPROVAL_REQUIRED/OWNER_ACTION_REQUIRED (not only AWAITING_APPROVAL)."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    with kb.write_txn(conn):
        kb._append_event(
            conn, umbrella, kb.HANDOFF_EVENT_KIND,
            {
                "marker": kb.HANDOFF_MARKER,
                "gate_id": gate,
                "verdict": "ACCEPTED",
                "repo_state": {},
                "next_action": {"type": "DELIVERY_CHECKPOINT", "requiresApproval": True},
                "decision": {
                    "actionType": "DELIVERY_CHECKPOINT",
                    "decisionClass": kb.APPROVAL_REQUIRED,
                    "requiresApproval": True,
                    "actionStatus": kb.ACTION_STATUS_AWAITING_APPROVAL,
                    "ownerAction": "REQUIRED",
                },
                "autoContinue": False,
                "ownerAction": "REQUIRED",
                "discussion": {"status": kb.DISCUSSION_OWNER_ACTION_REQUIRED},
            },
        )
    u = kb.get_task(conn, umbrella)
    g = kb.get_task(conn, gate)
    assert u is not None and g is not None and u.completed_at and g.completed_at
    terminal_at = max(int(u.completed_at), int(g.completed_at))

    assert kb.cleanup_terminal_probe_missions(
        conn, now=int(terminal_at) + 10_000
    ) == []
    assert kb.get_task(conn, umbrella).status == "done"
    assert kb.get_task(conn, gate).status == "done"


def test_cleanup_legacy_stale_starting_signature_still_archives(conn):
    """Non-regression (Security re-review point 3): a legacy_derived probe
    whose newest handoff IS the stale-STARTING signature (STARTING + window
    elapsed) is still auto-archived after the window."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    # Real-looking umbrella but the STARTING handoff was never materialized —
    # the exact signature of a stuck legacy probe (assignee NULL).
    _plant_legacy_starting_handoff(conn, umbrella, gate)
    planted_at = _newest_handoff_created_at(conn, umbrella)

    assert kb.cleanup_terminal_probe_missions(
        conn, window_seconds=120, now=planted_at + 119
    ) == []
    assert kb.get_task(conn, umbrella).status == "done"
    assert kb.cleanup_terminal_probe_missions(
        conn, window_seconds=120, now=planted_at + 120
    ) == [umbrella]
    assert kb.get_task(conn, umbrella).status == "archived"
    assert kb.get_task(conn, gate).status == "archived"


def test_cleanup_mission_kind_probe_still_archives_after_window(conn):
    """Non-regression: a STRUCTURED mission_kind='probe' mission (never
    legacy-derived) is still auto-archived after the bounded window regardless
    of the handoff actionStatus."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    conn.execute(
        "UPDATE tasks SET mission_kind = 'probe' WHERE id = ?", (umbrella,)
    )
    # Structured probes may legitimately carry an owner-approval handoff after
    # a gate; the marker (not the handoff) is the authorization.
    _plant_owner_approval_handoff(conn, umbrella, gate)
    planted_at = _newest_handoff_created_at(conn, umbrella)

    assert kb.cleanup_terminal_probe_missions(
        conn, window_seconds=120, now=planted_at + 119
    ) == []
    assert kb.cleanup_terminal_probe_missions(
        conn, window_seconds=120, now=planted_at + 120
    ) == [umbrella]
    assert kb.get_task(conn, umbrella).status == "archived"
    assert kb.get_task(conn, gate).status == "archived"


def test_cleanup_window_applies_to_any_action_status(conn):
    """SEC-PROBE-002: the bounded window applies regardless of the newest
    handoff actionStatus (not only STARTING). A fresh AWAITING_APPROVAL handoff
    on a structured probe defers the archive until the window elapses."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    conn.execute(
        "UPDATE tasks SET mission_kind = 'probe' WHERE id = ?", (umbrella,)
    )
    _plant_owner_approval_handoff(conn, umbrella, gate)
    planted_at = _newest_handoff_created_at(conn, umbrella)

    # Still inside the bounded window -> no archive, whatever actionStatus.
    assert kb.cleanup_terminal_probe_missions(
        conn, window_seconds=120, now=planted_at + 1
    ) == []
    assert kb.get_task(conn, umbrella).status == "done"
    # Window elapsed -> structured probe is archived.
    assert kb.cleanup_terminal_probe_missions(
        conn, window_seconds=120, now=planted_at + 120
    ) == [umbrella]
    assert kb.get_task(conn, umbrella).status == "archived"


def test_cleanup_resumes_legacy_archive_when_archive_ready_already_journaled(conn):
    """SEC-PROBE-001 clause (c): an already-journaled ARCHIVE_READY is a
    decision recorded by a previous tick (crash between journal and effective
    archive). Even a legacy-derived umbrella whose newest handoff is
    AWAITING_APPROVAL may complete that decided archive — the journal, not
    the handoff, is the authorization on resume."""
    umbrella, gate = _run_mission(
        conn,
        verdict_result="ACCEPTED",
        verdict_summary="All checks green.",
    )
    # The umbrella is legacy-derived: NO assignee, NO mission_kind — but a
    # previous tick already journaled ARCHIVE_READY (archive interrupted).
    with kb.write_txn(conn):
        kb._append_event(
            conn, umbrella, kb.TERMINAL_PROBE_CLEANUP_EVENT_KIND,
            {
                "classification": kb.DISCUSSION_ARCHIVE_READY,
                "reason": "probe_terminal",
                "evidence": {"identity_source": "legacy_derived"},
            },
        )
    _plant_owner_approval_handoff(conn, umbrella, gate)

    assert kb.cleanup_terminal_probe_missions(conn) == [umbrella]
    u = kb.get_task(conn, umbrella)
    g = kb.get_task(conn, gate)
    assert u is not None and u.status == "archived"
    assert g is not None and g.status == "archived"
