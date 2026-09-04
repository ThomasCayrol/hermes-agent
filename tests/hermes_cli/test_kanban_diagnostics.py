"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48






def test_repeated_crashes_truncates_huge_tracebacks():
    """Full Python tracebacks can be tens of KB. The title stays one
    line (≤160 chars); the detail caps at 500 chars + ellipsis so the
    card doesn't explode visually."""
    huge = "Traceback (most recent call last):\n" + ("  File\n" * 500)
    task = _task(status="ready")
    runs = [
        _run(outcome="crashed", run_id=1, error=huge),
        _run(outcome="crashed", run_id=2, error=huge),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    d = diags[0]
    # Title only the first line, capped.
    assert "\n" not in d.title
    assert len(d.title) < 250
    # Detail contains the snippet with ellipsis.
    assert d.detail.endswith("…") or len(d.detail) < 700


# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True


# ---------------------------------------------------------------------------
# Operator Diagnostics Clarity — attention axis + stranded classifier A-H
# (Product contract t_40e1e279 §1-9 + UX contract t_4b6987a2 §7).
#
# The two-axis model: technical severity (warning/error/critical) is
# preserved; an additive OPERATOR ATTENTION axis (NONE/INFO/WARNING/
# ACTION_REQUIRED/CRITICAL) plus owner_action / system_action /
# attention_banner / auto_recovery_state / classification ride on every
# Diagnostic and are exposed in to_dict.
# ---------------------------------------------------------------------------


def _stranded_ctx(**overrides):
    """Default read-only board_context evidence for the classifier.

    Fail-safe by default: dispatcher health and assignee validity are only
    present when the caller actually observed them (absence != health).
    """
    ctx = {
        "profiles": ["demo"],
        "lanes": {"demo": True},
        "profile_cap": 1,
        "board_cap": None,
        "dispatcher": None,
        "running_by_assignee": {"demo": []},
        "queue_by_assignee": {"demo": []},
        "queue_progressed_by_assignee": {"demo": False},
        "expected_slot_freed_by_assignee": {"demo": False},
        "attempts_by_task": {},
        "active_scope_tasks": [],
        "out_of_band_writers": None,
        "superseded_scope_tasks": [],
    }
    ctx.update(overrides)
    return ctx


def _stranded_diag(task, events, *, ctx=None, now=100_000, expected_kind="stranded_in_ready"):
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now, board_context=ctx,
    )
    return [d for d in diags if d.kind == expected_kind]


def test_case_A_legitimately_queued_is_info_no_banner():
    """A: ready 45 min, profile cap 1/1 occupied by a healthy running
    sibling, dispatcher healthy, queue advancing -> LEGITIMATELY_QUEUED,
    attention INFO, owner action NONE, excluded from the attention banner;
    copy never says \"no worker\"."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [_event("created", ts=now - 45 * 60)]
    ctx = _stranded_ctx(
        dispatcher={"healthy": True, "last_tick_ts": now - 30, "board_impact": False},
        running_by_assignee={"demo": [{"id": "t_sib01", "stale": False}]},
        queue_by_assignee={
            "demo": [{"id": "t_demo00", "priority": 0}, {"id": "t_sib01", "priority": 0}],
        },
        queue_progressed_by_assignee={"demo": True},
    )
    stranded = _stranded_diag(task, events, ctx=ctx, now=now)
    assert len(stranded) == 1
    d = stranded[0]
    assert d.classification == kd.CLASSIFICATION_LEGITIMATELY_QUEUED
    assert d.attention == kd.ATTENTION_INFO
    assert d.owner_action == kd.OWNER_ACTION_NONE
    assert d.attention_banner is False
    assert d.auto_recovery_state == kd.AUTO_RECOVERY_NONE
    assert d.operator_status  # French operator copy present
    assert "no worker" not in d.operator_status.lower()
    assert "no worker" not in (d.title or "").lower()


def test_case_B_ready_too_long_unexplained_is_warning_banner():
    """B: ready 45 min, capacity available, no claim, assignee valid,
    dispatcher healthy, no legitimate explanation -> READY_TOO_LONG_UNEXPLAINED,
    attention WARNING, attention_banner True, never alarmist about the
    dispatcher (it is healthy)."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [_event("created", ts=now - 45 * 60)]
    ctx = _stranded_ctx(
        dispatcher={"healthy": True, "last_tick_ts": now - 30, "board_impact": False},
        running_by_assignee={"demo": []},  # capacity free
        queue_progressed_by_assignee={"demo": False},
    )
    stranded = _stranded_diag(task, events, ctx=ctx, now=now)
    assert len(stranded) == 1
    d = stranded[0]
    assert d.classification == kd.CLASSIFICATION_READY_TOO_LONG_UNEXPLAINED
    assert d.attention == kd.ATTENTION_WARNING
    assert d.attention_banner is True
    assert d.owner_action == kd.OWNER_ACTION_NONE  # WARNING = recommended, not blocking
    assert "dispatcher" not in (d.operator_cause or "").lower() or "sain" in (d.operator_cause or "").lower()
    kinds = [a.kind for a in d.actions]
    assert "run_diagnostics" in kinds and "reassign" in kinds
    cli = [a for a in d.actions if a.kind == "cli_hint"]
    assert all(not a.suggested for a in cli)  # raw CLI stays secondary


def test_case_C_dispatcher_unhealthy_action_required():
    """C: dispatcher dead / no recent tick -> DISPATCHER_UNHEALTHY,
    attention ACTION_REQUIRED (CRITICAL when board impact), banner True,
    primary button \"Diagnostiquer le dispatcher\"."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [_event("created", ts=now - 45 * 60)]
    ctx = _stranded_ctx(
        dispatcher={"healthy": False, "last_tick_ts": now - 3600, "board_impact": False},
    )
    stranded = _stranded_diag(task, events, ctx=ctx, now=now)
    assert len(stranded) == 1
    d = stranded[0]
    assert d.classification == kd.CLASSIFICATION_DISPATCHER_UNHEALTHY
    assert d.attention == kd.ATTENTION_ACTION_REQUIRED
    assert d.owner_action == kd.OWNER_ACTION_REQUIRED
    assert d.attention_banner is True
    assert d.severity == "error"
    run = [a for a in d.actions if a.kind == "run_diagnostics"]
    assert run and run[0].suggested and run[0].label == "Diagnostiquer le dispatcher"

    ctx_crit = _stranded_ctx(
        dispatcher={"healthy": False, "last_tick_ts": now - 3600, "board_impact": True},
    )
    d_crit = _stranded_diag(task, events, ctx=ctx_crit, now=now)[0]
    assert d_crit.attention == kd.ATTENTION_CRITICAL
    assert d_crit.severity == "critical"


def test_case_C_never_infers_dispatcher_unhealthy_from_age_alone():
    """Product rule: DISPATCHER_UNHEALTHY requires dispatcher evidence;
    a bare aged task (no dispatcher context) must NOT be labelled
    dispatcher-unhealthy (fail-safe defaults to unexplained)."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [_event("created", ts=now - 45 * 60)]
    stranded = _stranded_diag(task, events, ctx=None, now=now)
    assert len(stranded) == 1
    d = stranded[0]
    assert d.classification != kd.CLASSIFICATION_DISPATCHER_UNHEALTHY
    assert d.classification == kd.CLASSIFICATION_READY_TOO_LONG_UNEXPLAINED
    assert d.attention == kd.ATTENTION_WARNING
    assert d.attention_banner is True  # conservative fail-safe keeps the signal


def test_case_D_recovery_in_progress_info_and_failure_action_required():
    """D: auto-recovery lifecycle markers -> recovery_in_progress INFO
    (no banner) while running; recovery_failed -> ACTION_REQUIRED + banner.
    History is preserved (events untouched, marker action kept in data)."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 120),
        _event("recovery_started", ts=now - 60, action="stale_claim_reclaim"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    recovering = [d for d in diags if d.kind == "recovery_in_progress"]
    assert len(recovering) == 1
    d = recovering[0]
    assert d.attention == kd.ATTENTION_INFO
    assert d.owner_action == kd.OWNER_ACTION_NONE
    assert d.attention_banner is False
    assert d.auto_recovery_state == kd.AUTO_RECOVERY_IN_PROGRESS
    assert d.data.get("action") == "stale_claim_reclaim"
    assert len(events) == 2  # history untouched

    failed_events = events + [
        _event("recovery_failed", ts=now - 30, action="stale_claim_reclaim"),
    ]
    diags_f = kd.compute_task_diagnostics(task, failed_events, [], now=now)
    failed = [d for d in diags_f if d.kind == "recovery_failed"]
    assert len(failed) == 1
    d_f = failed[0]
    assert d_f.attention == kd.ATTENTION_ACTION_REQUIRED
    assert d_f.owner_action == kd.OWNER_ACTION_REQUIRED
    assert d_f.attention_banner is True
    assert d_f.auto_recovery_state == kd.AUTO_RECOVERY_FAILED
    assert any(a.kind == "reclaim" for a in d_f.actions)
    assert not any(d.kind == "recovery_in_progress" for d in diags_f)


def test_case_D_recovery_succeeded_clears():
    """A successful recovery marker after a start clears the signal
    (auto-clearing diagnostics: no active recovery state to surface)."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 120),
        _event("recovery_started", ts=now - 60, action="stale_claim_reclaim"),
        _event("recovery_succeeded", ts=now - 30, action="stale_claim_reclaim"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert not any(d.kind in ("recovery_in_progress", "recovery_failed") for d in diags)


def test_case_E_approval_required_prominent_action_required():
    """E: task blocked for an owner decision (approval required) -> immediate
    ACTION_REQUIRED with REQUIRED ACTION + WHY surfaced in data, banner True
    (no 24h staleness wait for an explicit approval decision)."""
    now = int(time.time())
    task = _task(status="blocked", assignee="demo")
    events = [
        _event("blocked", ts=now - 600, reason="approval-required: recovery needs owner approval",
               decision_class="APPROVAL_REQUIRED"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stuck = [d for d in diags if d.kind == "stuck_in_blocked"]
    assert len(stuck) == 1
    d = stuck[0]
    assert d.attention == kd.ATTENTION_ACTION_REQUIRED
    assert d.owner_action == kd.OWNER_ACTION_REQUIRED
    assert d.attention_banner is True
    assert d.data.get("decision_class") == "APPROVAL_REQUIRED"
    assert d.data.get("required_action")
    assert d.data.get("why")
    kinds = [a.kind for a in d.actions]
    assert "comment" in kinds
    assert "approbation" in (d.operator_status or "").lower()


def test_case_F_long_running_fresh_heartbeat_no_warning():
    """F: worker running a long time with a fresh heartbeat -> NO warning at
    all (honest RUNNING; never a fabricated stalled/STARTING state)."""
    now = int(time.time())
    task = _task(
        status="running", assignee="demo",
        claim_lock="demo:run", claim_expires=now + 300,
        worker_pid=4242, last_heartbeat_at=now - 5, started_at=now - 3 * 3600,
    )
    events = [_event("claimed", ts=now - 3 * 3600, source_status="ready")]
    runs = [{"id": 1, "status": "running", "outcome": None,
             "started_at": now - 3 * 3600, "ended_at": None}]
    diags = kd.compute_task_diagnostics(task, events, runs, now=now)
    assert diags == []


def test_case_G_duplicate_and_concurrent_writer_diagnostics():
    """G: another active task on the same repo+branch+scope ->
    duplicate_implementation (owner decision unless supersession proven);
    an out-of-band (non-kanban) writer on the same checkout ->
    concurrent_writer_risk. STATUS/CAUSE/RISK/OWNER ACTION payload."""
    now = int(time.time())
    base = dict(
        status="running", assignee="lead", claim_lock="lead:1",
        workspace_path="/repo", branch_name="feat/x", project_id="p1",
    )
    task = _task(**base)
    events = [_event("claimed", ts=now - 600, source_status="ready")]
    ctx = {
        "active_scope_tasks": [
            {"id": "t_other", "status": "running", "assignee": "lead",
             "workspace_path": "/repo", "branch_name": "feat/x", "project_id": "p1"},
        ],
        "out_of_band_writers": [{"checkout": "/repo", "source": "external-daemon"}],
        "superseded_scope_tasks": [],
    }
    diags = kd.compute_task_diagnostics(task, events, [], now=now, board_context=ctx)
    dup = [d for d in diags if d.kind == "duplicate_implementation"]
    assert len(dup) == 1
    d = dup[0]
    assert d.attention == kd.ATTENTION_ACTION_REQUIRED
    assert d.owner_action == kd.OWNER_ACTION_REQUIRED
    assert d.attention_banner is True
    assert d.data.get("other_task_ids") == ["t_other"]
    assert d.operator_risk  # RISK replaces IMPACT for concurrency kinds

    con = [d for d in diags if d.kind == "concurrent_writer_risk"]
    assert len(con) == 1
    c = con[0]
    assert c.attention == kd.ATTENTION_ACTION_REQUIRED
    assert c.owner_action == kd.OWNER_ACTION_REQUIRED
    assert c.attention_banner is True
    assert c.severity == "error"
    assert c.operator_risk


def test_case_G_supersession_proven_is_warning_not_blocking():
    """Auto-consolidation is only signalled when supersession is proven:
    a proven-superseded duplicate is not ACTION_REQUIRED (no owner block)."""
    now = int(time.time())
    task = _task(
        status="running", assignee="lead", claim_lock="lead:1",
        workspace_path="/repo", branch_name="feat/x", project_id="p1",
    )
    ctx = {
        "active_scope_tasks": [
            {"id": "t_keep", "status": "running", "assignee": "lead",
             "workspace_path": "/repo", "branch_name": "feat/x", "project_id": "p1"},
        ],
        "superseded_scope_tasks": ["t_demo00"],
        "out_of_band_writers": None,
    }
    diags = kd.compute_task_diagnostics(
        task, [_event("claimed", ts=now - 600)], [], now=now, board_context=ctx,
    )
    dup = [d for d in diags if d.kind == "duplicate_implementation"]
    assert len(dup) == 1
    assert dup[0].attention != kd.ATTENTION_ACTION_REQUIRED
    assert dup[0].attention_banner is False


def test_case_G_concurrency_silent_without_board_context():
    """Without board context evidence the engine stays silent on concurrency
    (read-only detection; absence of evidence is not a finding)."""
    now = int(time.time())
    task = _task(
        status="running", assignee="lead", claim_lock="lead:1",
        workspace_path="/repo", branch_name="feat/x",
    )
    diags = kd.compute_task_diagnostics(task, [_event("claimed", ts=now - 600)], [], now=now)
    assert not any(d.kind in ("duplicate_implementation", "concurrent_writer_risk") for d in diags)


def test_case_H_action_buttons_real_actions_cli_secondary():
    """H: the emitted action contract — primary actions describe the real
    resulting action (run_diagnostics \"Diagnostiquer le dispatcher\" or
    reassign/reclaim), and the raw CLI command remains a secondary
    non-suggested affordance only."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [_event("created", ts=now - 45 * 60)]
    ctx = _stranded_ctx(
        dispatcher={"healthy": False, "last_tick_ts": now - 3600, "board_impact": False},
    )
    stranded = _stranded_diag(task, events, ctx=ctx, now=now)
    d = stranded[0]
    primary = [a for a in d.actions if a.suggested]
    assert primary and primary[0].kind == "run_diagnostics"
    assert all(a.kind != "cli_hint" for a in primary)
    cli = [a for a in d.actions if a.kind == "cli_hint"]
    if cli:
        assert not any(a.suggested for a in cli)
    for a in d.actions:
        if a.kind == "run_diagnostics":
            assert a.label == "Diagnostiquer le dispatcher"


def test_to_dict_exposes_operator_attention_axis():
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [_event("created", ts=now - 45 * 60)]
    ctx = _stranded_ctx(dispatcher={"healthy": False, "last_tick_ts": now - 3600, "board_impact": False})
    stranded = _stranded_diag(task, events, ctx=ctx, now=now)
    payload = stranded[0].to_dict()
    for key in (
        "attention", "owner_action", "system_action", "attention_banner",
        "auto_recovery_state", "classification",
        "operator_status", "operator_cause", "operator_impact",
    ):
        assert key in payload, f"to_dict missing {key}"
    assert payload["classification"] == kd.CLASSIFICATION_DISPATCHER_UNHEALTHY


def test_attention_banner_policy_matches_product_rule_5():
    assert kd.attention_banner_policy(attention="INFO") is False
    assert kd.attention_banner_policy(attention="ACTION_REQUIRED") is True
    assert kd.attention_banner_policy(attention="CRITICAL") is True
    assert kd.attention_banner_policy(attention="WARNING") is True  # abnormal + non-auto-recoverable default
    assert kd.attention_banner_policy(attention="WARNING", abnormal=False) is False
    assert kd.attention_banner_policy(attention="WARNING", auto_recoverable=True) is False
    assert kd.attention_banner_policy(attention="INFO", auto_recovery_state="failed") is True
    assert kd.owner_action_for_attention("ACTION_REQUIRED") == "REQUIRED"
    assert kd.owner_action_for_attention("CRITICAL") == "REQUIRED"
    assert kd.owner_action_for_attention("WARNING") == "NONE"
    assert kd.owner_action_for_attention("INFO") == "NONE"


def test_existing_kinds_carry_operator_attention_defaults():
    """Every diagnostic kind exposes an attention axis (matrix §4): actionable
    kinds land ACTION_REQUIRED + banner; informational ones INFO/no banner."""
    now = int(time.time())
    # hallucinated_cards is ACTION_REQUIRED + banner.
    task = _task(status="ready", assignee="demo")
    events = [
        _event("completion_blocked_hallucination", ts=now - 60,
               phantom_cards=["t_deadbeef1"]),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    hall = [d for d in diags if d.kind == "hallucinated_cards"]
    assert len(hall) == 1
    assert hall[0].attention == kd.ATTENTION_ACTION_REQUIRED
    assert hall[0].owner_action == kd.OWNER_ACTION_REQUIRED
    assert hall[0].attention_banner is True

    # prose_phantom_refs is informational: INFO, no banner.
    task2 = _task(status="ready", assignee="demo")
    events2 = [_event("suspected_hallucinated_references", ts=now - 60,
                      phantom_refs=["t_ghost1"])]
    diags2 = kd.compute_task_diagnostics(task2, events2, [], now=now)
    prose = [d for d in diags2 if d.kind == "prose_phantom_refs"]
    assert len(prose) == 1
    assert prose[0].attention == kd.ATTENTION_INFO
    assert prose[0].owner_action == kd.OWNER_ACTION_NONE
    assert prose[0].attention_banner is False


def test_stranded_cedes_to_repeated_failures():
    """Precedence anti-double-flag: when repeated failures already explain the
    task, the stranded rule does not also fire."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None,
                 consecutive_failures=3, last_failure_error="spawn boom")
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now, config={"failure_threshold": 2})
    assert not any(d.kind == "stranded_in_ready" for d in diags)
    assert any(d.kind == "repeated_failures" for d in diags)


def test_build_board_context_collects_running_and_queue(kanban_home):
    """The read-only board-context collector gathers running siblings, the
    ready queue, attempts and same-scope tasks from stable columns only."""
    conn = kb.connect()
    try:
        now = int(time.time())
        # One running task (fresh heartbeat) + two ready tasks for the same
        # assignee, plus a ready task for another assignee.
        run_id = kb.create_task(
            conn, title="running", assignee="demo", initial_status="running",
        )
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock='demo:1', claim_expires=?, "
            "worker_pid=7, last_heartbeat_at=?, started_at=? WHERE id = ?",
            (now + 300, now - 10, now - 100, run_id),
        )
        ready_a = kb.create_task(conn, title="ready a", assignee="demo")
        ready_b = kb.create_task(conn, title="ready b", assignee="demo")
        other = kb.create_task(conn, title="other", assignee="qa")
        conn.execute(
            "UPDATE tasks SET status='ready' WHERE id IN (?, ?, ?)",
            (ready_a, ready_b, other),
        )
        conn.commit()

        ctx = kd.build_board_context(
            conn, config={"kanban": {"max_in_progress_per_profile": 1}}, now=now,
        )
        assert ctx["profile_cap"] == 1
        running = ctx["running_by_assignee"].get("demo", [])
        assert len(running) == 1
        assert running[0]["id"] == run_id
        assert running[0]["stale"] is False
        ready_ids = {q["id"] for q in ctx["queue_by_assignee"].get("demo", [])}
        assert {ready_a, ready_b} <= ready_ids
        qa_ids = {q["id"] for q in ctx["queue_by_assignee"].get("qa", [])}
        assert other in qa_ids
        attempts = ctx["attempts_by_task"]
        assert attempts == {}  # no task_runs were created in this scenario
        scope_ids = {t["id"] for t in ctx["active_scope_tasks"]}
        assert {run_id, ready_a, ready_b, other} <= scope_ids
    finally:
        conn.close()
