"""Focused A-J contract for evidence-based Kanban stall recovery."""

from __future__ import annotations

import json
import sqlite3
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


def _event_payloads(conn, task_id: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()
    return [json.loads(row["payload"]) if row["payload"] else {} for row in rows]


def _completed_gate(conn, *, result: str = "ACCEPTED") -> tuple[str, str]:
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


def test_a_worker_disappears_is_orphaned_and_reclaimed_auto(conn, monkeypatch):
    task_id = kb.create_task(conn, title="Lost worker", assignee="worker")
    claimed = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:test")
    assert claimed is not None
    conn.execute(
        "UPDATE tasks SET worker_pid = 999999, started_at = ? WHERE id = ?",
        (int(time.time()) - 120, task_id),
    )
    conn.commit()
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    result = kb.dispatch_once(conn, dry_run=True)

    assert task_id in result.crashed
    event = _event_payloads(conn, task_id, "crashed")[-1]
    assert event["healthClassification"] == kb.ORPHANED
    assert event["decisionClass"] == kb.AUTO
    assert event["actionStatus"] == kb.ACTION_STATUS_RECOVERING
    assert event["ownerAction"] == "NONE"
    assert event["operatorMessage"].startswith("RÉCUPÉRATION AUTO")
    assert "READY-TO-SEND PROMPT" not in event["operatorMessage"]


def test_b_completed_gate_without_handoff_is_classified_and_recomputed_auto(conn):
    umbrella, gate = _completed_gate(conn)
    evidence = kb.task_recovery_evidence(conn, gate)

    health = kb.classify_task_health(evidence)
    emitted = kb.emit_terminal_handoffs_if_due(conn)

    assert health["classification"] == kb.COMPLETED_UNHANDED
    assert health["decisionClass"] == kb.AUTO
    assert health["nextAction"] == "RECOMPUTE_HANDOFF"
    assert emitted == [gate]
    handoff = _event_payloads(conn, umbrella, kb.HANDOFF_EVENT_KIND)[-1]
    assert handoff["healthClassification"] == kb.COMPLETED_UNHANDED


def test_c_proven_false_superseded_chain_is_reconciled_and_gate_rerun_auto(conn):
    umbrella, gate = _completed_gate(conn, result="REJECTED")
    false_v7 = kb.create_task(conn, title="v7 redacted false blocker", assignee="qa")
    dependent = kb.create_task(
        conn,
        title="Current dependent",
        assignee="lead",
        parents=[false_v7],
    )
    old_run = kb.claim_task(conn, false_v7)
    assert old_run is not None
    assert kb.complete_task(
        conn,
        false_v7,
        summary="Historical v7 result containing redacted output ***.",
        metadata={"verdict": "REJECTED"},
        expected_run_id=old_run.current_run_id,
    )
    before_runs = [run.id for run in kb.list_runs(conn, false_v7)]
    proof = {
        "kind": "direct_predicates",
        "source": "checkout_predicate",
        "redacted_output_used": False,
        "checks": [
            {
                "parent_id": false_v7,
                "child_id": dependent,
                "invalid": True,
                "superseded": True,
                "predicate": "literal_placeholder_absent",
                "observed": True,
            }
        ],
    }

    result = kb.reconcile_proven_superseded_dependencies(
        conn,
        proof=proof,
        superseded_task_ids=[false_v7],
        rerun_gate_id=gate,
        max_actions=4,
    )

    assert result["classification"] == kb.INVALID_SUPERSEDED_DEPENDENCY
    assert result["decisionClass"] == kb.AUTO
    assert result["actionStatus"] == kb.ACTION_STATUS_RECOVERING
    assert result["ownerAction"] == "NONE"
    assert result["unlinked"] == [[false_v7, dependent]]
    assert result["archived"] == [false_v7]
    assert result["rerunGate"] == gate
    assert kb.parent_ids(conn, dependent) == []
    assert kb.get_task(conn, false_v7).status == "archived"
    assert [run.id for run in kb.list_runs(conn, false_v7)] == before_runs
    assert kb.get_task(conn, gate).status == "ready"
    assert kb.get_task(conn, dependent).status == "ready"
    assert "***" not in json.dumps(result)
    recovery = _event_payloads(conn, gate, kb.RECOVERY_EVENT_KIND)[-1]
    assert recovery["operatorMessage"].startswith("RÉCUPÉRATION AUTO")
    assert "READY-TO-SEND PROMPT" not in recovery["operatorMessage"]

    replay = kb.reconcile_proven_superseded_dependencies(
        conn,
        proof=proof,
        superseded_task_ids=[false_v7],
        rerun_gate_id=gate,
        max_actions=4,
    )
    assert replay == result
    assert len(_event_payloads(conn, gate, kb.RECOVERY_EVENT_KIND)) == 1

    # A replay is only idempotent while the persisted result still matches.
    # Drift must fail closed rather than returning a stale success receipt.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done' WHERE id = ?",
            (false_v7,),
        )
        conn.execute(
            "INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)",
            (false_v7, dependent),
        )
    drifted_replay = kb.reconcile_proven_superseded_dependencies(
        conn,
        proof=proof,
        superseded_task_ids=[false_v7],
        rerun_gate_id=gate,
        max_actions=4,
    )
    assert drifted_replay["decisionClass"] == kb.APPROVAL_REQUIRED
    assert drifted_replay["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert kb.get_task(conn, false_v7).status == "done"
    assert false_v7 in kb.parent_ids(conn, dependent)


def test_c_missing_direct_proof_fails_closed_without_mutation(conn):
    _umbrella, gate = _completed_gate(conn, result="REJECTED")
    blocker = kb.create_task(conn, title="Unproven blocker", assignee="qa")
    child = kb.create_task(conn, title="Child", assignee="lead", parents=[blocker])

    result = kb.reconcile_proven_superseded_dependencies(
        conn,
        proof={"kind": "redacted_output", "source": "***", "checks": []},
        superseded_task_ids=[blocker],
        rerun_gate_id=gate,
    )

    assert result["classification"] == kb.STATE_INCONSISTENCY
    assert result["decisionClass"] == kb.APPROVAL_REQUIRED
    assert result["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert kb.parent_ids(conn, child) == [blocker]
    assert kb.get_task(conn, blocker).status != "archived"
    assert kb.get_task(conn, gate).status == "done"


def test_c_unproven_sibling_edge_fails_closed_without_mutation(conn):
    _umbrella, gate = _completed_gate(conn, result="REJECTED")
    blocker = kb.create_task(conn, title="Partially proven blocker", assignee="qa")
    proven_child = kb.create_task(
        conn, title="Proven child", assignee="lead", parents=[blocker]
    )
    unproven_child = kb.create_task(
        conn, title="Unproven child", assignee="lead", parents=[blocker]
    )
    proof = {
        "kind": "direct_predicates",
        "source": "checkout_predicate",
        "redacted_output_used": False,
        "checks": [{
            "parent_id": blocker,
            "child_id": proven_child,
            "invalid": True,
            "superseded": True,
            "predicate": "literal_placeholder_absent",
            "observed": True,
        }],
    }

    result = kb.reconcile_proven_superseded_dependencies(
        conn,
        proof=proof,
        superseded_task_ids=[blocker],
        rerun_gate_id=gate,
    )

    assert result["classification"] == kb.STATE_INCONSISTENCY
    assert result["decisionClass"] == kb.APPROVAL_REQUIRED
    assert kb.parent_ids(conn, proven_child) == [blocker]
    assert kb.parent_ids(conn, unproven_child) == [blocker]
    assert kb.get_task(conn, blocker).status != "archived"
    assert kb.get_task(conn, gate).status == "done"


def test_c_reconciliation_holds_write_lock_while_validating(conn, monkeypatch):
    _umbrella, gate = _completed_gate(conn, result="REJECTED")
    blocker = kb.create_task(conn, title="Race candidate", assignee="qa")
    child = kb.create_task(
        conn, title="Race dependent", assignee="lead", parents=[blocker]
    )
    # Keep the lock-probe fast: the second connection would otherwise sit in
    # SQLite's long default busy_timeout before surfacing "database is locked".
    conn.execute("PRAGMA busy_timeout=250")
    proof = {
        "kind": "direct_predicates",
        "source": "checkout_predicate",
        "redacted_output_used": False,
        "checks": [{
            "parent_id": blocker,
            "child_id": child,
            "invalid": True,
            "superseded": True,
            "predicate": "checkout_card_absent",
            "observed": True,
        }],
    }
    db_path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "250")
    other = kb.connect(Path(db_path))
    original_get_task = kb.get_task
    race = {"attempted": False, "blocked": False}

    def get_task_with_race(target_conn, task_id):
        task = original_get_task(target_conn, task_id)
        if task_id == blocker and not race["attempted"]:
            race["attempted"] = True
            try:
                with kb.write_txn(other):
                    other.execute(
                        "UPDATE tasks SET role = 'umbrella' WHERE id = ?",
                        (blocker,),
                    )
            except sqlite3.OperationalError:
                race["blocked"] = True
        return task

    monkeypatch.setattr(kb, "get_task", get_task_with_race)
    try:
        result = kb.reconcile_proven_superseded_dependencies(
            conn,
            proof=proof,
            superseded_task_ids=[blocker],
            rerun_gate_id=gate,
        )
    finally:
        other.close()

    assert result["decisionClass"] == kb.AUTO
    assert race == {"attempted": True, "blocked": True}
    assert original_get_task(conn, blocker).role != "umbrella"


def test_d_failed_task_is_retryable_auto(conn):
    task_id = kb.create_task(conn, title="QA failed", assignee="qa")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    conn.execute(
        "UPDATE task_runs SET status='failed', outcome='failed', ended_at=? WHERE id=?",
        (int(time.time()), claimed.current_run_id),
    )
    conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
        "claim_expires=NULL, worker_pid=NULL WHERE id=?",
        (task_id,),
    )
    conn.commit()

    health = kb.classify_task_health(kb.task_recovery_evidence(conn, task_id))

    assert health["classification"] == kb.FAILED
    assert health["decisionClass"] == kb.AUTO
    assert health["nextAction"] == "WORKER_RETRY"


def test_e_merge_ready_pr_requires_approval():
    action = kb.resolve_next_action(
        {
            "verdict": True,
            "active_workers": [],
            "blockers": [],
            "repo_state": {
                "branch": "feat/stall-recovery",
                "dirty": False,
                "committed": True,
                "pushed": True,
            },
        }
    )
    assert action["decisionClass"] == kb.APPROVAL_REQUIRED
    assert action["requiresApproval"] is True


def test_f_destructive_or_canonical_actions_require_approval():
    for action in ("MERGE_MAIN", "FORCE_PUSH", "DELETE_KANBAN_EVIDENCE"):
        decision = kb.classify_next_action(action)
        assert decision["decisionClass"] == kb.APPROVAL_REQUIRED
    assert kb.classify_next_action("COMMIT", branch="main")["decisionClass"] == kb.APPROVAL_REQUIRED


def test_g_approval_handoff_emits_exact_french_operator_headings(conn):
    umbrella, gate = _completed_gate(conn, result="ACCEPTED")
    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]
    comment = kb.list_comments(conn, umbrella)[-1].body
    headings = [
        "MISSION ARRÊTÉE",
        "Cause",
        "Étape bloquée",
        "Pourquoi Hermes ne peut pas continuer seul",
        "Solution recommandée",
        "Impact si aucune action",
        "RECOMMENDED OWNER DECISION",
        "WHY",
        "READY-TO-SEND PROMPT",
        "DECISION CLASS",
        "ACTION STATUS",
    ]
    positions = [comment.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "DECISION CLASS: APPROVAL_REQUIRED" in comment
    assert "ACTION STATUS: AWAITING_APPROVAL" in comment


def test_h_auto_handoff_without_execution_witness_fails_closed(conn):
    umbrella, gate = _completed_gate(conn, result="REJECTED")
    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]
    comment = kb.list_comments(conn, umbrella)[-1].body
    assert "MISSION ARRÊTÉE" in comment
    assert "OWNER ACTION: REQUIRED" in comment
    assert "ACTION STATUS: AWAITING_APPROVAL" in comment
    assert "Auto-continuation not persisted" in comment
    assert "READY-TO-SEND PROMPT" in comment


def test_i_historical_reject_prose_does_not_override_structured_accept(conn):
    umbrella, gate = _completed_gate(conn, result="ACCEPTED")
    conn.execute(
        "UPDATE task_runs SET summary=? WHERE task_id=? AND outcome='completed'",
        ("Security REJECTED v7; final structured verdict is authoritative.", gate),
    )
    conn.commit()
    task = kb.get_task(conn, gate)
    assert task is not None
    assert kb.gate_verdict(conn, task) is True
    assert kb.emit_terminal_handoffs_if_due(conn) == [gate]
    handoff = _event_payloads(conn, umbrella, kb.HANDOFF_EVENT_KIND)[-1]
    assert handoff["verdict"] == "ACCEPTED"


def test_j_legitimate_long_running_activity_is_not_stalled_by_elapsed_time(conn, monkeypatch):
    task_id = kb.create_task(conn, title="Long encoding", assignee="worker")
    claimed = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:long")
    assert claimed is not None
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET worker_pid=4242, started_at=?, last_heartbeat_at=NULL WHERE id=?",
        (now - 20_000, task_id),
    )
    conn.execute(
        "UPDATE task_runs SET started_at=?, last_heartbeat_at=NULL WHERE id=?",
        (now - 20_000, claimed.current_run_id),
    )
    conn.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at, run_id) "
        "VALUES (?, 'activity', ?, ?, ?)",
        (task_id, json.dumps({"source": "tool_progress"}), now - 5, claimed.current_run_id),
    )
    conn.commit()
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)

    reclaimed = kb.detect_stale_running(conn, stale_timeout_seconds=100)
    health = kb.classify_task_health(
        kb.task_recovery_evidence(conn, task_id, now=now),
        now=now,
        stale_timeout_seconds=100,
    )

    assert reclaimed == []
    assert health["classification"] == kb.ACTIVE
    assert kb.get_task(conn, task_id).status == "running"
