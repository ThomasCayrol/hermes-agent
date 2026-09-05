"""Operator-clarity contract for observable Kanban recovery states."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _completed_gate(conn, *, result: str) -> tuple[str, str]:
    umbrella = kb.create_task(conn, title="Mission", role="umbrella")
    gate = kb.create_task(
        conn,
        title="Final gate",
        role="gate",
        parents=[umbrella],
        assignee="gate-owner",
    )
    assert kb.complete_task(conn, umbrella, summary="mission prepared")
    run = kb.claim_task(conn, gate)
    assert run is not None
    assert kb.complete_task(
        conn,
        gate,
        result=result,
        summary="Final gate completed.",
        metadata={"verdict": result},
        expected_run_id=run.current_run_id,
    )
    return umbrella, gate


def _last_payload(conn, task_id: str, kind: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()
    assert row is not None
    return json.loads(row["payload"] or "{}")


def test_a_live_worker_with_fresh_heartbeat_is_running_and_needs_no_owner_action(conn):
    task_id = kb.create_task(conn, title="Active worker", assignee="worker")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, last_heartbeat_at = ? WHERE id = ?",
            (os.getpid(), now, task_id),
        )

    state = kb.classify_task_health(
        kb.task_recovery_evidence(conn, task_id, now=now),
        now=now,
        stale_timeout_seconds=1,
    )

    assert state["classification"] == kb.ACTIVE
    assert state["actionStatus"] == kb.ACTION_STATUS_RUNNING
    assert state["ownerAction"] == "NONE"
    assert state["runningEvidence"]["workerPid"] == os.getpid()
    assert state["runningEvidence"]["runId"] == claimed.current_run_id
    assert state["runningEvidence"]["lastHeartbeat"] == now
    assert state["runningEvidence"]["currentTask"] == task_id


def test_b_auto_next_action_without_execution_fails_closed_to_approval(conn):
    umbrella, gate = _completed_gate(conn, result="REJECTED")

    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]

    comment = kb.list_comments(conn, umbrella)[-1].body
    payload = _last_payload(conn, umbrella, kb.HANDOFF_EVENT_KIND)
    assert "OWNER ACTION: REQUIRED" in comment
    assert "ACTION STATUS: AWAITING_APPROVAL" in comment
    assert "ACTION AUTO PLANIFIÉE" not in comment
    assert "Auto-continuation not persisted" in comment
    assert payload["decision"]["ownerAction"] == "REQUIRED"
    assert payload["decision"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert payload["decision"]["failClosedToApproval"] is True


def test_c_approval_immediately_exposes_required_action_and_why(conn):
    umbrella, gate = _completed_gate(conn, result="ACCEPTED")

    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]

    comment = kb.list_comments(conn, umbrella)[-1].body
    owner = comment.index("OWNER ACTION: REQUIRED")
    required = comment.index("REQUIRED ACTION:", owner)
    why = comment.index("WHY:", required)
    assert owner < required < why
    assert "ACTION STATUS: AWAITING_APPROVAL" in comment
    payload = _last_payload(conn, umbrella, kb.HANDOFF_EVENT_KIND)
    assert payload["decision"]["ownerAction"] == "REQUIRED"
    assert payload["decision"]["requiredAction"]
    assert payload["decision"]["why"]


def test_d_active_bounded_recovery_is_recovering_and_needs_no_owner_action(
    conn, monkeypatch,
):
    task_id = kb.create_task(conn, title="Lost worker", assignee="worker")
    claimed = kb.claim_task(
        conn,
        task_id,
        claimer=f"{kb._claimer_id().split(':', 1)[0]}:operator-clarity",
    )
    assert claimed is not None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
            (999_999, int(time.time()) - 120, task_id),
        )
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    dispatched = kb.dispatch_once(conn, dry_run=True)

    assert task_id in dispatched.crashed
    payload = _last_payload(conn, task_id, "crashed")
    assert payload["actionStatus"] == kb.ACTION_STATUS_RECOVERING
    assert payload["ownerAction"] == "NONE"


def test_e_completed_gate_awaiting_merge_requires_owner_approval(
    conn, monkeypatch,
):
    umbrella, gate = _completed_gate(conn, result="ACCEPTED")
    monkeypatch.setattr(
        kb,
        "repo_state_for",
        lambda _task: {
            "branch": "feat/operator-clarity",
            "dirty": False,
            "committed": True,
            "pushed": True,
            "unpushed_commits": 0,
        },
    )

    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]

    comment = kb.list_comments(conn, umbrella)[-1].body
    payload = _last_payload(conn, umbrella, kb.HANDOFF_EVENT_KIND)
    assert payload["next_action"]["type"] == "INTEGRATION_REVIEW"
    assert payload["decision"]["ownerAction"] == "REQUIRED"
    assert payload["decision"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert "OWNER ACTION: REQUIRED" in comment
    assert "ACTION STATUS: AWAITING_APPROVAL" in comment


def test_f_long_running_handoff_without_execution_witness_fails_closed(conn):
    umbrella, gate = _completed_gate(conn, result="REJECTED")
    worker = kb.create_task(
        conn,
        title="Long-running encoder",
        assignee="worker",
        parents=[umbrella],
    )
    claimed = kb.claim_task(conn, worker)
    assert claimed is not None
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = ?, last_heartbeat_at = ? "
            "WHERE id = ?",
            (os.getpid(), now - 20_000, now, worker),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ?, last_heartbeat_at = ? WHERE id = ?",
            (now - 20_000, now, claimed.current_run_id),
        )

    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]

    comment = kb.list_comments(conn, umbrella)[-1].body
    assert "OWNER ACTION: REQUIRED" in comment
    assert "ACTION STATUS: AWAITING_APPROVAL" in comment
    assert "RUNNING EVIDENCE" not in comment
    assert "Auto-continuation not persisted" in comment
