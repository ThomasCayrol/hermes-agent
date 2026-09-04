"""Tests for the Hermes autonomous next-action policy.

Covers (mission spec 2026-09-03, "explicit Hermes autonomous next-action
policy"):

  A. rejected gate -> remediation AUTO
  B. remediation pass -> re-review AUTO
  C. QA rejection -> remediation AUTO
  D. accepted gate -> safe delivery preparation AUTO (feature branch)
  E. merge main -> APPROVAL_REQUIRED
  F. destructive migration -> APPROVAL_REQUIRED
  G. force push -> APPROVAL_REQUIRED
  H. branch-local commit -> AUTO
  I. ambiguous product-direction choice -> APPROVAL_REQUIRED
  J. ACCEPT verdict containing historical word REJECT -> still ACCEPT
  K. restart/replay does not duplicate auto-launched workflow
  L. same nextAction is idempotent across gateway restart

plus verdict-derivation precedence (structured metadata verdict ->
task.result -> carefully parsed legacy text fallback) and the persisted
operator autonomy policy (kanban.autonomy config).
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


def _run_gate_mission(conn, *, result=None, summary=None, metadata=None):
    """Create + finish a canonical mission (umbrella -> done gate)."""
    umbrella = kb.create_task(conn, title="Mission", role="umbrella")
    gate = kb.create_task(conn, title="Gate", role="gate", parents=[umbrella])
    kb.claim_task(conn, umbrella)
    kb.complete_task(conn, umbrella, result="mission underway")
    assert kb.claim_task(conn, gate) is not None
    ok = kb.complete_task(
        conn, gate,
        result=result,
        summary=summary,
        metadata=metadata,
    )
    assert ok
    return umbrella, gate


def _handoff_comments(conn, task_id) -> list:
    return [
        c.body for c in kb.list_comments(conn, task_id) if kb.HANDOFF_MARKER in c.body
    ]


def _events(conn, task_id, kind) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()
    out = []
    for row in rows:
        out.append(json.loads(row["payload"]) if row["payload"] else {})
    return out


def _newest_handoff(conn, task_id) -> dict:
    events = _events(conn, task_id, kb.HANDOFF_EVENT_KIND)
    assert events, "no terminal_handoff event recorded"
    return events[-1]


# ---------------------------------------------------------------------------
# Verdict derivation precedence (spec §1 + scenario J)
# ---------------------------------------------------------------------------


def test_j_accept_verdict_containing_historical_reject_is_accept(conn):
    """Regression: 'Security REJECT' history must not flip a structured
    ACCEPTED result or a summary that opens with the gate's own ACCEPT line."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="ACCEPTED",
        summary=(
            "Security REJECTED the first attempt; remediation landed; final gate "
            "accepted everything. All green."
        ),
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.gate_verdict(conn, gate_task) is True


def test_verdict_precedence_metadata_over_result_over_summary(conn):
    """Layer 1 (run metadata verdict) beats layer 2 (task.result) which beats
    layer 3 (legacy summary text)."""
    # metadata ACCEPTED vs result REJECTED + summary REJECT -> ACCEPTED
    _, g1 = _run_gate_mission(
        conn,
        result="REJECTED",
        summary="Gate rejected: blockers open.",
        metadata={"verdict": "ACCEPTED"},
    )
    task1 = kb.get_task(conn, g1)
    assert task1 is not None
    assert kb.gate_verdict(conn, task1) is True

    # metadata REJECTED vs result ACCEPTED + summary ACCEPT -> REJECTED
    _, g2 = _run_gate_mission(
        conn,
        result="ACCEPTED",
        summary="All checks green.",
        metadata={"verdict": "REJECTED"},
    )
    task2 = kb.get_task(conn, g2)
    assert task2 is not None
    assert kb.gate_verdict(conn, task2) is False


def test_verdict_precedence_structured_result_over_summary(conn):
    """Layer 2 (canonical task.result, parsed in isolation) beats layer 3 —
    conflicting prose never contaminates the structured result channel."""
    _, g_accept = _run_gate_mission(
        conn,
        result="ACCEPTED",
        summary="QA REJECTED the candidate; changes required.",
    )
    task_accept = kb.get_task(conn, g_accept)
    assert task_accept is not None
    assert kb.gate_verdict(conn, task_accept) is True

    _, g_reject = _run_gate_mission(
        conn,
        result="REJECT",
        summary="QA accepted on first pass; nothing left to fix.",
    )
    task_reject = kb.get_task(conn, g_reject)
    assert task_reject is not None
    assert kb.gate_verdict(conn, task_reject) is False


def test_verdict_metadata_json_string_and_verdicts_list(conn):
    """metadata may arrive as a JSON string (sqlite column), may carry a
    unanimous ``verdicts`` list, and real gate runs persist the structured
    ``gate_outcome`` field."""
    assert kb._verdict_from_run_metadata('{"verdict": "ACCEPTED"}') is True
    assert kb._verdict_from_run_metadata('{"verdict": "CHANGES_REQUIRED"}') is False
    assert kb._verdict_from_run_metadata({"verdicts": ["ACCEPTED", "PASS"]}) is True
    assert kb._verdict_from_run_metadata({"verdicts": ["ACCEPTED", "REJECTED"]}) is None
    assert kb._verdict_from_run_metadata({"summary": "ACCEPTED here"}) is None
    assert kb._verdict_from_run_metadata("not json {") is None
    assert kb._verdict_from_run_metadata(None) is None
    # Real gate runs persist gate_outcome (e.g. the AppStock final gate).
    assert kb._verdict_from_run_metadata({"gate_outcome": "ACCEPT"}) is True
    assert kb._verdict_from_run_metadata({"gate_outcome": "REJECT"}) is False
    assert kb._verdict_from_run_metadata(
        {"gate_outcome": "REJECT", "verdict": "ACCEPTED"}
    ) is True  # explicit verdict key outranks the alias


def test_verdict_canonical_token_not_substring(conn):
    """Only exact canonical tokens are structured verdicts — never a
    substring of prose."""
    assert kb._canonical_verdict("ACCEPTED") is True
    assert kb._canonical_verdict("reject") is False
    assert kb._canonical_verdict("NOT ACCEPTED") is False
    assert kb._canonical_verdict("changes_required") is False
    assert kb._canonical_verdict("ACCEPTED SOON") is None
    assert kb._canonical_verdict("This gate was ACCEPTED.") is None


def test_verdict_legacy_text_only_evidence(conn):
    """Text-only legacy evidence: the last completed run's opening verdict
    line is decisive when unambiguous."""
    _, g1 = _run_gate_mission(
        conn,
        result=None,
        summary="CDP FINAL GATE ACCEPT — accepted after all evidence verified.",
    )
    task1 = kb.get_task(conn, g1)
    assert task1 is not None
    assert kb.gate_verdict(conn, task1) is True

    _, g2 = _run_gate_mission(
        conn,
        result=None,
        summary="QA REJECTED — FAIL-1/2/3 open, changes required.",
    )
    task2 = kb.get_task(conn, g2)
    assert task2 is not None
    assert kb.gate_verdict(conn, task2) is False

    _, g3 = _run_gate_mission(
        conn,
        result=None,
        summary="Neutral narrative without any verdict statement.",
    )
    task3 = kb.get_task(conn, g3)
    assert task3 is not None
    assert kb.gate_verdict(conn, task3) is None


def test_verdict_ambiguous_free_text_fails_closed(conn):
    """Mixed accept+reject evidence in a free-text opening is ambiguous:
    None (fail closed toward a human decision), never an arbitrary substring
    guess."""
    _, g1 = _run_gate_mission(
        conn,
        result=None,
        summary=(
            "QA accepted the remediation work; the earlier REJECT findings "
            "were all closed."
        ),
    )
    task1 = kb.get_task(conn, g1)
    assert task1 is not None
    assert kb.gate_verdict(conn, task1) is None


# ---------------------------------------------------------------------------
# Decision classification (spec §2–§4 + scenarios A–I)
# ---------------------------------------------------------------------------


def _decision(action_type, **kw):
    out = kb.classify_next_action(action_type, **kw)
    assert set(out) == {"decisionClass", "requiresApproval", "rationale"}
    return out


def test_a_rejected_gate_resolves_remediation_auto():
    action = kb.resolve_next_action({
        "verdict": False, "active_workers": [], "blockers": [],
        "repo_state": {"branch": "fix/x"},
    })
    assert action["type"] == "REMEDIATION"
    assert action["decisionClass"] == kb.AUTO
    assert action["requiresApproval"] is False
    assert action["rationale"]


def test_b_rerun_reviews_are_auto():
    for t in ("RE_RUN_SECURITY_REVIEW", "RE_RUN_DEVOPS_REVIEW",
              "RE_RUN_QA", "RE_RUN_FINAL_GATE", "RE_RUN_REVIEW"):
        assert _decision(t)["decisionClass"] == kb.AUTO


def test_c_qa_rejection_resolves_remediation_auto():
    # QA rejection surfaces as a REJECTED gate verdict -> REMEDIATION AUTO.
    action = kb.resolve_next_action({
        "verdict": False, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": True, "branch": "fix/x"},
    })
    assert action["type"] == "REMEDIATION"
    assert action["decisionClass"] == kb.AUTO


def test_d_accepted_dirty_feature_branch_delivery_auto():
    auto = kb.resolve_next_action({
        "verdict": True, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": True, "committed": True, "pushed": False,
                       "branch": "feat/x"},
    })
    assert auto["type"] == "DELIVERY_CHECKPOINT"
    assert auto["decisionClass"] == kb.AUTO
    assert auto["requiresApproval"] is False

    required = kb.resolve_next_action({
        "verdict": True, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": True, "committed": True, "pushed": False,
                       "branch": "main"},
    })
    assert required["type"] == "DELIVERY_CHECKPOINT"
    assert required["decisionClass"] == kb.APPROVAL_REQUIRED
    assert required["requiresApproval"] is True


def test_e_merge_main_requires_approval():
    assert _decision("MERGE_MAIN")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("MERGE")["decisionClass"] == kb.APPROVAL_REQUIRED
    # Accepted + pushed -> INTEGRATION_REVIEW whose milestone is the merge
    # decision: approval required.
    action = kb.resolve_next_action({
        "verdict": True, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": False, "committed": True, "pushed": True,
                       "branch": "feat/x"},
    })
    assert action["type"] == "INTEGRATION_REVIEW"
    assert action["decisionClass"] == kb.APPROVAL_REQUIRED


def test_f_destructive_migration_requires_approval():
    assert _decision("DESTRUCTIVE_MIGRATION")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("DESTRUCTIVE_DB_MIGRATION")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("IRREVERSIBLE_PRODUCTION_OP")["decisionClass"] == kb.APPROVAL_REQUIRED


def test_g_force_push_requires_approval():
    assert _decision("FORCE_PUSH")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("HISTORY_REWRITE")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("PUSH_MAIN")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("DELETE_BRANCH")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("DELETE_KANBAN_EVIDENCE")["decisionClass"] == kb.APPROVAL_REQUIRED


def test_h_branch_local_commit_is_auto():
    assert _decision("BRANCH_COMMIT")["decisionClass"] == kb.AUTO
    assert _decision("COMMIT", branch="feat/x")["decisionClass"] == kb.AUTO
    assert _decision("COMMIT", branch="main")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("COMMIT", branch="develop",
                     policy={"protectedBranches": ["develop"]})["decisionClass"] == kb.APPROVAL_REQUIRED
    # No confirmable repository identity -> fail closed.
    assert _decision("COMMIT")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("PUSH", branch="feat/x")["decisionClass"] == kb.AUTO
    assert _decision("PUSH", branch="main")["decisionClass"] == kb.APPROVAL_REQUIRED


def test_i_ambiguous_product_direction_requires_approval():
    assert _decision("PRODUCT_DIRECTION_CHOICE")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("PRODUCT_SCOPE_EXPANSION")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("ARCHITECTURE_REDESIGN")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("SECURITY_POLICY_CHANGE")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("PRODUCTION_DEPLOYMENT")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("EXTERNAL_PUBLICATION")["decisionClass"] == kb.APPROVAL_REQUIRED


def test_unknown_action_fails_closed():
    assert _decision("SOMETHING_IMAGINATIVE")["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision(None)["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("")["decisionClass"] == kb.APPROVAL_REQUIRED


def test_low_risk_internal_actions_are_auto():
    for t in ("TESTS", "RUN_TESTS", "AUDIT", "BROWSER_VERIFICATION",
              "KANBAN_CARD_CREATE", "KANBAN_CARD_UPDATE",
              "CREATE_FEATURE_BRANCH", "DOCUMENTATION",
              "WORKER_RETRY", "EPHEMERAL_CLEANUP"):
        assert _decision(t)["decisionClass"] == kb.AUTO, t


def test_resolver_waiting_states_classified():
    blocker = kb.resolve_next_action({
        "verdict": True, "active_workers": [], "blockers": ["t_b1"],
        "repo_state": {},
    })
    assert blocker["decisionClass"] == kb.APPROVAL_REQUIRED

    active = kb.resolve_next_action({
        "verdict": True, "active_workers": ["t_run (hotelos-lead)"],
        "blockers": [], "repo_state": {},
    })
    assert active["decisionClass"] == kb.AUTO

    unknown = kb.resolve_next_action({
        "verdict": None, "active_workers": [], "blockers": [],
        "repo_state": {},
    })
    assert unknown["decisionClass"] == kb.APPROVAL_REQUIRED


# ---------------------------------------------------------------------------
# Persisted operator autonomy policy (spec §5)
# ---------------------------------------------------------------------------


def test_autonomy_policy_from_config_defaults():
    policy = kb.autonomy_policy_from_config({})
    assert policy == kb.DEFAULT_AUTONOMY_POLICY
    assert policy["mode"] == kb.AUTONOMY_MODE_AUTO


def test_autonomy_policy_from_config_normalises():
    policy = kb.autonomy_policy_from_config({
        "kanban": {
            "autonomy": {
                "mode": "checkpoint",
                "approvals": {"REMEDIATION": "auto", "MERGE_MAIN": "maybe"},
                "protectedBranches": ["develop", ""],
            }
        }
    })
    assert policy["mode"] == "checkpoint"
    assert policy["approvals"] == {"REMEDIATION": "auto"}
    assert policy["protectedBranches"] == ["develop"]
    # invalid mode falls back to auto
    policy2 = kb.autonomy_policy_from_config({
        "kanban": {"autonomy": {"mode": "banana"}}
    })
    assert policy2["mode"] == kb.AUTONOMY_MODE_AUTO


def test_checkpoint_mode_requires_approval_everywhere():
    policy = kb.autonomy_policy_from_config({"kanban": {"autonomy": {"mode": "checkpoint"}}})
    assert _decision("REMEDIATION", policy=policy)["decisionClass"] == kb.APPROVAL_REQUIRED
    assert _decision("DOCUMENTATION", policy=policy)["decisionClass"] == kb.APPROVAL_REQUIRED


def test_per_action_override_wins_over_mode_and_tables():
    # required override forces even an otherwise-auto action to stop.
    required_policy = kb.autonomy_policy_from_config({
        "kanban": {"autonomy": {"approvals": {"REMEDIATION": "required"}}}
    })
    assert _decision("REMEDIATION", policy=required_policy)["decisionClass"] == kb.APPROVAL_REQUIRED
    # auto override permits even an otherwise-approval action (explicit waiver).
    auto_policy = kb.autonomy_policy_from_config({
        "kanban": {"autonomy": {"approvals": {"MERGE_MAIN": "auto"}}}
    })
    assert _decision("MERGE_MAIN", policy=auto_policy)["decisionClass"] == kb.AUTO


def test_resolve_next_action_accepts_operator_policy():
    policy = kb.autonomy_policy_from_config({"kanban": {"autonomy": {"mode": "checkpoint"}}})
    action = kb.resolve_next_action({
        "verdict": False, "active_workers": [], "blockers": [],
        "repo_state": {},
    }, autonomy_policy=policy)
    assert action["type"] == "REMEDIATION"
    assert action["decisionClass"] == kb.APPROVAL_REQUIRED


def test_resolve_next_action_accepts_legacy_approvals_policy():
    # Back-compat: {verb: bool} approvals_policy maps True -> required.
    action = kb.resolve_next_action({
        "verdict": True, "active_workers": [], "blockers": [],
        "repo_state": {"dirty": True, "branch": "feat/x"},
    }, approvals_policy={"commit": True})
    assert action["type"] == "DELIVERY_CHECKPOINT"
    assert action["decisionClass"] == kb.APPROVAL_REQUIRED


# ---------------------------------------------------------------------------
# Terminal-handoff decision block + auto-continuation (spec §6–§7)
# ---------------------------------------------------------------------------


def test_handoff_rejected_gate_records_planned_auto_remediation(conn):
    """AUTO authorizes remediation without claiming execution has begun."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="REJECT",
        summary="QA rejected the delivery: FAIL-1 open.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(conn, gate_task) is True

    comments = _handoff_comments(conn, umbrella)
    assert len(comments) == 1
    assert "Decision class: AUTO" in comments[0]
    assert "ACTION AUTO PLANIFIÉE" in comments[0]
    assert "OWNER ACTION: NONE" in comments[0]
    assert "ACTION STATUS: STARTING" in comments[0]
    assert "started automatically" not in comments[0]
    assert "Analysis:" in comments[0]

    newest = _newest_handoff(conn, umbrella)
    decision = newest["decision"]
    assert decision["actionType"] == "REMEDIATION"
    assert decision["decisionClass"] == kb.AUTO
    assert decision["actionStatus"] == kb.ACTION_STATUS_STARTING
    assert decision["ownerAction"] == "NONE"
    assert decision["requiresApproval"] is False
    assert decision["rationale"]
    assert decision["analysis"].startswith("Final gate")
    assert newest["autoContinue"] is True

    continue_events = _events(conn, umbrella, kb.AUTO_CONTINUE_EVENT_KIND)
    assert len(continue_events) == 1
    assert continue_events[0]["actionType"] == "REMEDIATION"
    assert continue_events[0]["gate_id"] == gate


def test_handoff_ambiguous_verdict_never_auto(conn):
    """No explicit verdict -> AWAITING_DECISION APPROVAL_REQUIRED; no
    auto_continue event is recorded."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="unrelated note",
        summary="QA accepted the work; earlier REJECT findings were closed.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    # Ambiguous/mixed prose -> gate_verdict None.
    assert kb.gate_verdict(conn, gate_task) is None
    assert kb.emit_terminal_handoff(conn, gate_task) is True

    newest = _newest_handoff(conn, umbrella)
    assert newest["verdict"] == "NO EXPLICIT VERDICT"
    assert newest["decision"]["decisionClass"] == kb.APPROVAL_REQUIRED
    assert newest["decision"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    assert _events(conn, umbrella, kb.AUTO_CONTINUE_EVENT_KIND) == []


def test_handoff_accepted_mainline_delivery_requires_approval(conn, tmp_path):
    """ACCEPTED + dirty tree on main -> DELIVERY_CHECKPOINT APPROVAL_REQUIRED
    (canonical-state mutation), action status AWAITING_APPROVAL, no
    auto_continue."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    (repo / "a.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    (repo / "b.txt").write_text("y")  # dirty on main

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
    assert kb.emit_terminal_handoff(conn, gate_task) is True

    newest = _newest_handoff(conn, umbrella)
    assert newest["verdict"] == "ACCEPTED"
    assert newest["next_action"]["type"] == "DELIVERY_CHECKPOINT"
    assert newest["decision"]["decisionClass"] == kb.APPROVAL_REQUIRED
    assert newest["decision"]["actionStatus"] == kb.ACTION_STATUS_AWAITING_APPROVAL
    comments = _handoff_comments(conn, umbrella)
    assert "Decision class: APPROVAL_REQUIRED" in comments[0]
    assert "Awaiting your approval" in comments[0]
    assert _events(conn, umbrella, kb.AUTO_CONTINUE_EVENT_KIND) == []


def test_k_restart_replay_does_not_duplicate_auto_launch(conn):
    """Restart/replay of an AUTO handoff never duplicates the handoff or its
    recorded auto_continue event."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="REJECT",
        summary="Gate rejected: changes required.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None
    assert kb.emit_terminal_handoff(conn, gate_task) is True
    assert len(_events(conn, umbrella, kb.HANDOFF_EVENT_KIND)) == 1
    assert len(_events(conn, umbrella, kb.AUTO_CONTINUE_EVENT_KIND)) == 1

    # Brand-new connection (restarted gateway / orchestrator replay).
    with kb.connect() as conn2:
        assert kb.emit_terminal_handoffs_if_due(conn2) == []
        assert len(_events(conn2, umbrella, kb.HANDOFF_EVENT_KIND)) == 1
        assert len(_events(conn2, umbrella, kb.AUTO_CONTINUE_EVENT_KIND)) == 1
        assert len(_handoff_comments(conn2, umbrella)) == 1


def test_l_same_next_action_idempotent_across_gateway_restart(conn):
    """Repeated dispatcher scans (across connections) resolve the same
    nextAction and emit exactly one handoff with a stable decision."""
    umbrella, gate = _run_gate_mission(
        conn,
        result="REJECT",
        summary="QA rejected — remediation required.",
    )
    gate_task = kb.get_task(conn, gate)
    assert gate_task is not None

    with kb.connect() as conn1:
        assert kb.emit_terminal_handoffs_if_due(conn1) == [gate]
    with kb.connect() as conn2:
        # Second gateway generation sees no due gates and no new emission.
        assert kb.emit_terminal_handoffs_if_due(conn2) == []

    newest = _newest_handoff(conn, umbrella)
    assert newest["decision"]["actionType"] == "REMEDIATION"
    assert newest["decision"]["decisionClass"] == kb.AUTO
    # The persisted decision is stable across observers: re-derive from the
    # same persisted facts and compare identity fields.
    gate_task_live = kb.get_task(conn, gate)
    umbrella_task_live = kb.get_task(conn, umbrella)
    assert gate_task_live is not None and umbrella_task_live is not None
    snapshot = kb.synthesize_terminal_handoff(
        conn, gate_task_live, umbrella_task_live
    )
    rederived = kb.resolve_next_action(snapshot)
    assert rederived["type"] == newest["decision"]["actionType"]
    assert rederived["decisionClass"] == newest["decision"]["decisionClass"]
