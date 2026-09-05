"""Tests for hermes_cli.kanban_external_ci — the external-CI wait watchdog.

Maps the Product acceptance matrix A-J (t_80b64bdb §6) plus the transverse
evidence/config/idempotency rules:

  A. queued 10 min no steps          -> CI_WAITING, NO alert (silent)
  B. queued > warning no steps       -> CI_WAITING_LONG (WARNING, KEEP_OPEN)
  C. queued > stall no steps         -> CI_INFRA_STALLED (prominent; never a
                                        worker stall)
  D. job running / steps started     -> NOT stalled (RUNNING natural)
  E. completed failure after real
     execution                       -> CI_FAILED, never CI_INFRA_STALLED
  F. safe AUTO retry accepted (2xx)  -> CI_RECOVERING
  G. after retry, step starts        -> RUNNING; checks pass -> COMPLETED
  H. retry refused 403 / queued past
     bounded window                  -> OWNER_ACTION_REQUIRED; never a retry
                                        loop, never an auto-cancel
  I. superseding workflow run exists -> old queued run never false-alerts
  J. Mission Control stays read-only (this module never writes outside the
     kanban event/comment rows it emits; no MC schema/UI change here)

Plus transverse rules: no stall classification without a complete snapshot
(fail-closed), started_at non-null is NOT execution proof, config defaults /
mission overrides / invalid config fallback, dominance order, idempotent
emission (1x per state + evidence set), and the 10-field operator schema.

The classifier itself is pure (no DB). Emission/evaluation tests use a real
in-memory kanban DB via the shared ``kanban_home`` fixture style.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_external_ci as kc


# ---------------------------------------------------------------------------
# Fixtures + builders
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _job(job_id, *, status="queued", conclusion=None, steps=None, started_at=None,
         run_id=1, name="Run tests"):
    return {
        "id": job_id,
        "run_id": run_id,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": started_at,
        "completed_at": None,
        "steps": steps if steps is not None else [],
        "html_url": f"https://github.com/o/r/actions/runs/{run_id}/job/{job_id}",
    }


def _run(run_id, *, created_at, status="queued", conclusion=None, head_sha="sha1"):
    return {
        "id": run_id,
        "status": status,
        "conclusion": conclusion,
        "created_at": created_at,
        "head_sha": head_sha,
    }


def _snapshot(
    *,
    repo="ThomasCayrol/hermes-agent",
    pr_number=6,
    head_sha="sha1",
    runs=None,
    jobs=None,
    captured_at=None,
    required=True,
    superseded=False,
):
    now = int(captured_at if captured_at is not None else time.time())
    runs = runs if runs is not None else [_run(1, created_at=now - 600)]
    jobs = jobs if jobs is not None else [_job(101, started_at=now - 600)]
    return kc.EvidenceSnapshot(
        captured_at=now,
        repo=repo,
        pr_number=pr_number,
        head_sha=head_sha,
        runs=runs,
        jobs=jobs,
        required=required,
        superseded=superseded,
    )


def _queued_since(minutes: int, job_id: int = 101, run_id: int = 1) -> kc.EvidenceSnapshot:
    """A snapshot whose required job has been queued (steps=0) for ``minutes``."""
    now = int(time.time())
    queued_at = now - minutes * 60
    return _snapshot(
        runs=[_run(run_id, created_at=queued_at)],
        jobs=[_job(job_id, started_at=queued_at, run_id=run_id)],
        captured_at=now,
    )


def _create_mission(conn, task_id="t_ci0001", role="umbrella"):
    return kb.create_task(conn, title="CI mission", role=role, idempotency_key=task_id)


# ---------------------------------------------------------------------------
# A-J acceptance matrix (classifier is pure)
# ---------------------------------------------------------------------------


def test_A_queued_10min_no_steps_is_waiting_no_alert():
    snap = _queued_since(10)
    result = kc.classify_external_ci_wait(snap, now=snap.captured_at)
    assert result["ci_state"] == kc.CI_WAITING
    assert result["evidence_ok"] is True
    assert result["queue_minutes"] == 10
    alert = kc.build_operator_alert(result, pr_url="https://github.com/o/r/pull/6")
    assert alert["attention"] == "NONE"
    assert alert["discussionStatus"] == "KEEP_OPEN"
    assert alert["ownerAction"] == "NONE"


def test_B_queued_over_warning_no_steps_is_waiting_long():
    snap = _queued_since(48)  # > 45 warning default
    result = kc.classify_external_ci_wait(snap, now=snap.captured_at)
    assert result["ci_state"] == kc.CI_WAITING_LONG
    alert = kc.build_operator_alert(result, pr_url="")
    assert alert["attention"] == "WARNING"
    assert alert["discussionStatus"] == "KEEP_OPEN"
    assert alert["ownerAction"] == "NONE"
    assert "Aucun test n'a encore démarré" in alert["summary"]
    assert "48m" in alert["summary"]


def test_C_queued_over_stall_no_steps_is_infra_stalled():
    snap = _queued_since(84)  # real PR #6 incident: > 60 stall default
    result = kc.classify_external_ci_wait(snap, now=snap.captured_at)
    assert result["ci_state"] == kc.CI_INFRA_STALLED
    alert = kc.build_operator_alert(result, pr_url="")
    assert alert["attention"] == "ACTION_REQUIRED"
    assert "Aucun échec de code n'est observé" in alert["summary"]
    assert "Relancer uniquement les jobs queued concernés" in alert["recommendedAction"]
    # Never a worker-stall vocabulary.
    assert "worker stall" not in alert["summary"].lower()
    assert "reclaim" not in alert["summary"].lower()


def test_D_job_in_progress_or_steps_started_is_not_stalled():
    # Even if the total queue is long, an in_progress job means execution.
    now = int(time.time())
    queued_at = now - 3000
    snap = _snapshot(
        runs=[_run(1, created_at=queued_at, status="in_progress")],
        jobs=[_job(101, status="in_progress", steps=[{"name": "s1"}], started_at=now - 600)],
        captured_at=now,
    )
    result = kc.classify_external_ci_wait(snap, now=now)
    assert result["ci_state"] == kc.RUNNING
    assert result["ci_state"] != kc.CI_INFRA_STALLED


def test_D2_started_at_nonnull_is_not_execution_proof():
    """Queued jobs carry started_at = enqueue time — steps==0 must NOT read
    as running, and a >stall queued job with non-null started_at IS a stall."""
    now = int(time.time())
    queued_since = 4000  # > 60m stall default
    snap = _snapshot(
        runs=[_run(1, created_at=now - queued_since)],
        jobs=[_job(101, status="queued", started_at=now - queued_since, steps=[])],
        captured_at=now,
    )
    # started_at non-null alone is NOT execution proof.
    assert not kc.job_execution_started(snap.jobs[0])
    assert not kc.job_is_running(snap.jobs[0])
    result = kc.classify_external_ci_wait(snap, now=now)
    assert result["ci_state"] == kc.CI_INFRA_STALLED  # real stall (>60m, steps=0)
    assert result["ci_state"] != kc.RUNNING


def test_E_completed_failure_after_execution_is_failed_not_stalled():
    now = int(time.time())
    snap = _snapshot(
        runs=[_run(1, created_at=now - 600, status="completed", conclusion="failure")],
        jobs=[
            _job(
                101, status="completed", conclusion="failure",
                steps=[{"name": "s1", "conclusion": "failure"}],
                started_at=now - 500,
            )
        ],
        captured_at=now,
    )
    result = kc.classify_external_ci_wait(snap, now=now)
    assert result["ci_state"] == kc.CI_FAILED
    assert result["ci_state"] != kc.CI_INFRA_STALLED
    alert = kc.build_operator_alert(result, pr_url="")
    assert alert["ownerAction"] == "REQUIRED"
    assert "exécution" in alert["summary"] or "démarré" in alert["summary"]


def test_F_auto_retry_accepted_is_recovering(conn):
    task_id = _create_mission(conn)
    snap = _queued_since(90, job_id=777, run_id=42)

    def fake_runner(args):
        # Accept the job-level rerun POST (2xx) with an empty body.
        return 204, {}

    out = kc.evaluate_external_ci_wait(
        conn, task_id, snap,
        now=snap.captured_at,
        runner=fake_runner,
        config={"kanban": {"ci_wait": {}}},
    )
    assert out["retry"] is not None
    assert out["retry"]["accepted"] is True
    assert out["retry"]["level"] == kc.RETRY_JOB_LEVEL
    assert out["classification"]["ci_state"] == kc.CI_RECOVERING
    assert out["emitted"] is True
    # UX sequence: the prominent CI_INFRA_STALLED alert (OWNER ACTION NONE,
    # KEEP_OPEN) was emitted BEFORE the accepted retry updated to RECOVERING.
    assert out["stallAlertEmitted"] is True
    events = [
        e for e in kb.list_events(conn, task_id)
        if e.kind == kc.EXTERNAL_CI_WAIT_EVENT_KIND
    ]
    assert [(e.payload or {}).get("ciState") for e in events] == [
        kc.CI_INFRA_STALLED, kc.CI_RECOVERING,
    ]
    assert (events[0].payload or {}).get("ownerAction") == "NONE"
    assert (events[0].payload or {}).get("discussionStatus") == "KEEP_OPEN"
    # Retry attempt recorded for audit.
    assert kc.retry_attempted_for_episode(conn, task_id) is True


def test_G_after_recovering_step_starts_then_completed(conn):
    task_id = _create_mission(conn)
    # Phase 1: stalled -> AUTO retry accepted -> CI_RECOVERING.
    snap1 = _queued_since(90, job_id=777, run_id=42)

    def accept(args):
        return 204, {}

    kc.evaluate_external_ci_wait(
        conn, task_id, snap1, now=snap1.captured_at, runner=accept,
        config={"kanban": {"ci_wait": {}}},
    )
    # Phase 2: a step started (in_progress) -> RUNNING.
    now = int(time.time())
    snap2 = _snapshot(
        runs=[_run(42, created_at=now - 600, status="in_progress")],
        jobs=[
            _job(777, status="in_progress", steps=[{"name": "s1"}],
                 started_at=now - 400, run_id=42)
        ],
        captured_at=now,
    )
    out2 = kc.evaluate_external_ci_wait(
        conn, task_id, snap2, now=now, runner=accept,
        config={"kanban": {"ci_wait": {}}},
    )
    assert out2["classification"]["ci_state"] == kc.RUNNING
    # Phase 3: all checks passed -> COMPLETED.
    now3 = int(time.time())
    snap3 = _snapshot(
        runs=[_run(42, created_at=now - 600, status="completed", conclusion="success")],
        jobs=[
            _job(777, status="completed", conclusion="success",
                 steps=[{"name": "s1", "conclusion": "success"}],
                 started_at=now - 400, run_id=42)
        ],
        captured_at=now3,
    )
    out3 = kc.evaluate_external_ci_wait(
        conn, task_id, snap3, now=now3, runner=accept,
        config={"kanban": {"ci_wait": {}}},
    )
    assert out3["classification"]["ci_state"] == kc.COMPLETED


def test_H_retry_refused_403_is_owner_action_required(conn):
    task_id = _create_mission(conn)
    snap = _queued_since(90, job_id=777, run_id=42)

    def refuse(args):
        # Real-world finding: GitHub refuses rerun of queued jobs on an
        # ACTIVE run with 403 "workflow run ... already running".
        return 403, {"message": "workflow run 42 already running"}

    out = kc.evaluate_external_ci_wait(
        conn, task_id, snap,
        now=snap.captured_at,
        runner=refuse,
        config={"kanban": {"ci_wait": {}}},
    )
    assert out["retry"]["accepted"] is False
    assert out["retry"]["statusCode"] == 403
    assert out["classification"]["ci_state"] == kc.OWNER_ACTION_REQUIRED
    alert = out["alert"]
    assert alert["ownerAction"] == "REQUIRED"
    assert alert["requiredAction"]
    assert alert["why"]
    assert alert["discussionStatus"] == "OWNER_ACTION_REQUIRED"
    # No retry loop: a second evaluation must NOT re-attempt.
    out2 = kc.evaluate_external_ci_wait(
        conn, task_id, snap,
        now=snap.captured_at,
        runner=refuse,
        config={"kanban": {"ci_wait": {}}},
    )
    assert out2["retry"] is not None  # still shows the recorded refusal
    events = [
        e for e in kb.list_events(conn, task_id)
        if e.kind == kc.EXTERNAL_CI_RETRY_EVENT_KIND
    ]
    assert len(events) == 1  # ONE bounded attempt per episode


def test_H2_retry_queued_past_window_is_owner_action_required(conn):
    task_id = _create_mission(conn)
    # First tick: stalled -> accepted retry -> CI_RECOVERING.
    snap1 = _queued_since(90, job_id=777, run_id=42)
    kc.evaluate_external_ci_wait(
        conn, task_id, snap1, now=snap1.captured_at,
        runner=lambda args: (204, {}),
        config={"kanban": {"ci_wait": {"retry_window_minutes": 30}}},
    )
    # 40 minutes later the retried job is STILL queued (same episode, no
    # resolving state) -> the bounded window elapsed -> OWNER_ACTION_REQUIRED.
    later = snap1.captured_at + 40 * 60
    snap2 = _snapshot(
        runs=[_run(42, created_at=snap1.captured_at - 90 * 60)],
        jobs=[_job(777, started_at=snap1.captured_at - 90 * 60, run_id=42)],
        captured_at=later,
    )
    out2 = kc.evaluate_external_ci_wait(
        conn, task_id, snap2, now=later,
        runner=lambda args: (204, {}),
        config={"kanban": {"ci_wait": {"retry_window_minutes": 30}}},
    )
    assert out2["classification"]["ci_state"] == kc.OWNER_ACTION_REQUIRED
    assert out2["alert"]["ownerAction"] == "REQUIRED"
    # Still exactly one retry attempt (budget never loops).
    events = [
        e for e in kb.list_events(conn, task_id)
        if e.kind == kc.EXTERNAL_CI_RETRY_EVENT_KIND
    ]
    assert len(events) == 1


def test_I_superseded_run_never_false_alerts():
    now = int(time.time())
    queued_at = now - 5000  # way past stall
    snap = _snapshot(
        runs=[
            _run(2, created_at=now - 60, status="queued", head_sha="sha2"),
            _run(1, created_at=queued_at, status="queued", head_sha="sha1"),
        ],
        jobs=[_job(101, started_at=queued_at, run_id=1)],
        captured_at=now,
        superseded=True,
    )
    result = kc.classify_external_ci_wait(snap, now=now)
    assert result["ci_state"] == kc.CI_WAITING  # never alerts
    assert result["superseded"] is True
    alert = kc.build_operator_alert(result, pr_url="")
    assert alert["attention"] == "NONE"


def test_I2_pr_closed_no_longer_required_is_silent(conn):
    """A queued run whose PR is closed/merged is moot — no stall alert."""
    now = int(time.time())
    snap = _snapshot(
        runs=[_run(1, created_at=now - 5000, status="queued")],
        jobs=[_job(101, started_at=now - 5000)],
        captured_at=now,
        required=False,
    )
    result = kc.classify_external_ci_wait(snap, now=now)
    assert result["ci_state"] == kc.COMPLETED  # wait moot, silent
    task_id = _create_mission(conn)
    out = kc.evaluate_external_ci_wait(
        conn, task_id, snap, now=now,
        config={"kanban": {"ci_wait": {}}},
    )
    # No alert was ever emitted -> resolution stays silent.
    assert out["emitted"] is False


def test_J_mission_control_read_only_no_new_writes():
    """The watchdog's writes are kanban event/comment rows only.

    Mission Control stays read-only: the module never opens a second DB /
    board and never mutates task status or worker bookkeeping. Behavioural
    proof: a stalled evaluation leaves the task row untouched (covered in
    test_J2) and emits only the two documented event kinds.
    """
    assert kc.EXTERNAL_CI_WAIT_EVENT_KIND == "external_ci_wait"
    assert kc.EXTERNAL_CI_RETRY_EVENT_KIND == "external_ci_retry"


def test_J2_external_wait_never_mutates_task_status_or_worker_state(conn):
    """A CI_INFRA_STALLED evaluation must NOT reclaim/block/fail the task:
    an external-dependency wait is never a Hermes worker stall. The task
    stays in its normal execution status (KEEP_OPEN by construction)."""
    task_id = _create_mission(conn)
    assert kb.claim_task(conn, task_id) is not None  # task now running

    snap = _queued_since(90, job_id=555, run_id=5)
    kc.evaluate_external_ci_wait(
        conn, task_id, snap, now=snap.captured_at,
        auto_remediation=False,
        config={"kanban": {"ci_wait": {}}},
    )

    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"          # never blocked/reclaimed
    assert task.current_run_id is not None   # run still open
    events = {e.kind for e in kb.list_events(conn, task_id)}
    # Only comment/CI-watchdog events; never worker-recovery vocabulary.
    assert not (events & {"reclaimed", "blocked", "stale", "crashed", "timed_out"})


# ---------------------------------------------------------------------------
# Transverse rules: evidence fail-closed, config, dominance, idempotency
# ---------------------------------------------------------------------------


def test_fail_closed_empty_snapshot_no_stall_evidence_missing_signal(conn):
    task_id = _create_mission(conn)
    empty = kc.EvidenceSnapshot.from_dict({})
    result = kc.classify_external_ci_wait(empty)
    assert result["ci_state"] == kc.CI_WAITING
    assert result["evidence_ok"] is False
    assert result["ci_state"] != kc.CI_INFRA_STALLED
    # Evaluation emits a non-alarmist INFO evidence-missing signal, idempotent.
    out = kc.evaluate_external_ci_wait(
        conn, task_id, empty, now=int(time.time()),
        config={"kanban": {"ci_wait": {}}},
    )
    assert out["classification"]["evidence_ok"] is False
    assert out["classification"]["ci_state"] == kc.CI_WAITING
    assert out["emitted"] is True
    alert = out["alert"]
    assert alert["attention"] in ("INFO", "NONE")
    assert "Impossible de vérifier l'état GitHub" in alert["ciEvidence"]
    # Second identical evaluation -> no duplicate (idempotent fingerprint).
    out2 = kc.evaluate_external_ci_wait(
        conn, task_id, empty, now=int(time.time()),
        config={"kanban": {"ci_wait": {}}},
    )
    assert out2["emitted"] is False
    events = [
        e for e in kb.list_events(conn, task_id)
        if e.kind == kc.EXTERNAL_CI_WAIT_EVENT_KIND
    ]
    assert len(events) == 1


def test_A_silent_wait_emits_no_event(conn):
    """UX-1: CI_WAITING healthy produces ZERO visible artefacts."""
    task_id = _create_mission(conn)
    snap = _queued_since(10)
    out = kc.evaluate_external_ci_wait(
        conn, task_id, snap, now=snap.captured_at,
        config={"kanban": {"ci_wait": {}}},
    )
    assert out["emitted"] is False
    events = [e for e in kb.list_events(conn, task_id) if e.kind != "created"]
    assert events == []
    comments = kb.list_comments(conn, task_id)
    assert comments == []


def test_config_defaults_without_config():
    policy = kc.ci_wait_policy_from_config(None)
    assert policy["warning_minutes"] == 45
    assert policy["stall_minutes"] == 60
    assert policy["retry_window_minutes"] == 30
    assert policy["mission_overrides"] == {}


def test_config_invalid_falls_back_to_defaults():
    policy = kc.ci_wait_policy_from_config(
        {"kanban": {"ci_wait": {"warning_minutes": -3, "stall_minutes": "x"}}}
    )
    assert policy["warning_minutes"] == 45
    assert policy["stall_minutes"] == 60


def test_mission_override_applied_and_partial_inherits():
    policy = kc.ci_wait_policy_from_config(
        {
            "kanban": {
                "ci_wait": {
                    "warning_minutes": 45,
                    "stall_minutes": 60,
                    "mission_overrides": {
                        "t_umbrella01": {"stall_minutes": 120}
                    },
                }
            }
        }
    )
    default = kc.resolve_ci_wait_policy(policy)
    assert default["stall_minutes"] == 60
    overridden = kc.resolve_ci_wait_policy(policy, "t_umbrella01")
    assert overridden["stall_minutes"] == 120
    assert overridden["warning_minutes"] == 45  # partial override inherits


def test_warning_gte_stall_disables_waiting_long():
    policy = kc.ci_wait_policy_from_config(
        {"kanban": {"ci_wait": {"warning_minutes": 90, "stall_minutes": 60}}}
    )
    resolved = kc.resolve_ci_wait_policy(policy)
    assert resolved["warning_enabled"] is False
    snap = _queued_since(70)
    result = kc.classify_external_ci_wait(snap, now=snap.captured_at)
    # warning >= stall -> CI_WAITING_LONG never emitted; stall at 70m wins.
    assert result["ci_state"] == kc.CI_INFRA_STALLED


def test_dominance_failed_beats_stalled_beats_long_beats_waiting():
    def state_for(jobs):
        return kc.aggregate_ci_states([kc.classify_external_ci_wait(_snapshot(
            jobs=jobs,
            runs=[_run(1, created_at=int(time.time()) - 6000)],
            captured_at=int(time.time()),
        ))["ci_state"] for _ in [0]])

    assert state_for([_job(1, started_at=int(time.time()) - 6000)]) == kc.CI_INFRA_STALLED
    # A failure in one job dominates the aggregate even when another is queued.
    now = int(time.time())
    jobs_failed = [
        _job(1, started_at=now - 6000),
        _job(2, status="completed", conclusion="failure",
             steps=[{"name": "s1", "conclusion": "failure"}], started_at=now - 6000),
    ]
    c = kc.classify_external_ci_wait(_snapshot(jobs=jobs_failed, captured_at=now), now=now)
    assert c["ci_state"] == kc.CI_FAILED
    # Pure dominance helper ordering.
    assert kc.aggregate_ci_states(
        [kc.CI_WAITING_LONG, kc.CI_FAILED, kc.CI_INFRA_STALLED]
    ) == kc.CI_FAILED
    assert kc.aggregate_ci_states(
        [kc.CI_WAITING, kc.CI_WAITING_LONG, kc.CI_INFRA_STALLED]
    ) == kc.CI_INFRA_STALLED
    assert kc.aggregate_ci_states([kc.CI_WAITING, kc.COMPLETED]) == kc.CI_WAITING


def test_attention_owner_tokens_align_with_socle_diagnostics():
    """The alert payload reuses the existing operator tokens — no invented
    status vocabulary. Values must match kanban_diagnostics exactly."""
    from hermes_cli import kanban_diagnostics as kd

    assert kc.OWNER_ACTION_REQUIRED_TEXT == kd.OWNER_ACTION_REQUIRED
    allowed_attention = set(kd.ATTENTION_ORDER)
    for state in kc.CI_STATES:
        alert = kc.build_operator_alert({"ci_state": state})
        assert alert["attention"] in allowed_attention, state
        assert alert["ownerAction"] in (kd.OWNER_ACTION_NONE, kd.OWNER_ACTION_REQUIRED)
        assert alert["discussionStatus"] in ("KEEP_OPEN", "OWNER_ACTION_REQUIRED")
        assert alert["actionStatus"] in (
            "RUNNING", "FAILED", "RECOVERING", "COMPLETED", "AWAITING_APPROVAL",
        )


def test_alert_carries_10_field_schema_in_order():
    snap = _queued_since(70)
    result = kc.classify_external_ci_wait(snap, now=snap.captured_at)
    alert = kc.build_operator_alert(result, pr_url="https://github.com/o/r/pull/6")
    keys = list(alert.keys())
    schema = [
        "missionStatus", "summary", "externalDependencyStatus", "ciEvidence",
        "impact", "recommendedAction", "ownerAction", "nextAction",
        "discussionStatus", "discussionAction",
    ]
    idx = [keys.index(k) for k in schema if k in keys]
    assert idx == sorted(idx)
    for k in schema:
        assert k in alert
    # Rendered comment keeps the same canonical order.
    text = kc.render_alert_text(alert)
    positions = [
        text.index(label)
        for label in (
            "EXTERNAL DEPENDENCY STATUS", "MISSION STATUS", "SUMMARY",
            "CI EVIDENCE", "IMPACT", "RECOMMENDED ACTION", "OWNER ACTION",
            "NEXT ACTION", "DISCUSSION STATUS", "DISCUSSION ACTION",
        )
    ]
    assert positions == sorted(positions)


def test_emission_idempotent_per_state_and_jobset(conn):
    task_id = _create_mission(conn)
    # auto_remediation disabled: CI_INFRA_STALLED escalates to
    # OWNER_ACTION_REQUIRED (Hermes cannot retry) and that alert is emitted
    # ONCE — a second identical evaluation must not duplicate it.
    snap = _queued_since(70, job_id=111, run_id=1)
    kc.evaluate_external_ci_wait(
        conn, task_id, snap, now=snap.captured_at,
        auto_remediation=False,
        config={"kanban": {"ci_wait": {}}},
    )
    kc.evaluate_external_ci_wait(
        conn, task_id, snap, now=snap.captured_at,
        auto_remediation=False,
        config={"kanban": {"ci_wait": {}}},
    )
    events = [
        e for e in kb.list_events(conn, task_id)
        if e.kind == kc.EXTERNAL_CI_WAIT_EVENT_KIND
    ]
    assert len(events) == 1
    assert (events[0].payload or {}).get("ciState") == kc.OWNER_ACTION_REQUIRED
    # A different state (WAITING_LONG first, then STALLED->escalation) emits
    # fresh events (same-thread updates), not duplicates of the prior state.
    task2 = _create_mission(conn, task_id="t_ci0002")
    snap_long = _queued_since(48, job_id=222, run_id=2)
    kc.evaluate_external_ci_wait(
        conn, task2, snap_long, now=snap_long.captured_at,
        auto_remediation=False,
        config={"kanban": {"ci_wait": {}}},
    )
    snap_stalled = _queued_since(70, job_id=222, run_id=2)
    kc.evaluate_external_ci_wait(
        conn, task2, snap_stalled, now=snap_stalled.captured_at,
        auto_remediation=False,
        config={"kanban": {"ci_wait": {}}},
    )
    events2 = [
        e for e in kb.list_events(conn, task2)
        if e.kind == kc.EXTERNAL_CI_WAIT_EVENT_KIND
    ]
    assert len(events2) == 2
    assert (events2[0].payload or {}).get("ciState") == kc.CI_WAITING_LONG
    assert (events2[1].payload or {}).get("ciState") == kc.OWNER_ACTION_REQUIRED


def test_retry_result_never_simulates_recovery_on_5xx(conn):
    task_id = _create_mission(conn)
    snap = _queued_since(90, job_id=333, run_id=3)
    out = kc.evaluate_external_ci_wait(
        conn, task_id, snap, now=snap.captured_at,
        runner=lambda args: (500, {"message": "boom"}),
        config={"kanban": {"ci_wait": {}}},
    )
    assert out["retry"]["accepted"] is False
    assert out["classification"]["ci_state"] == kc.OWNER_ACTION_REQUIRED


def test_evidence_complete_requires_runs_jobs_head_sha():
    now = int(time.time())
    snap = kc.EvidenceSnapshot(
        captured_at=now, repo="o/r", pr_number=1, head_sha="", runs=[], jobs=[]
    )
    assert not snap.evidence_complete()
    snap2 = kc.EvidenceSnapshot(
        captured_at=now, repo="o/r", pr_number=1, head_sha="sha1",
        runs=[_run(1, created_at=now)], jobs=[_job(1, started_at=now)],
    )
    assert snap2.evidence_complete()


def test_run_external_ci_watchdog_for_pr_collects_and_evaluates(conn):
    """The one-shot activation hook collects live evidence then evaluates."""
    task_id = _create_mission(conn)
    now = int(time.time())
    queued_at = now - 4000

    def fake_gh(args):
        path = args[0] if isinstance(args, list) and args else ""
        if path.startswith("repos/o/r/pulls/6"):
            return 0, {
                "state": "open",
                "head": {"sha": "abc123"},
                "mergeable": True,
            }
        if "/actions/runs?" in path:
            # gh api ... --jq .workflow_runs returns the LIST.
            return 0, [
                {"id": 42, "head_sha": "abc123", "status": "queued",
                 "conclusion": None, "created_at": queued_at}
            ]
        if "/actions/runs/42/jobs" in path:
            return 0, [
                {"id": 777, "run_id": 42, "name": "Run tests",
                 "status": "queued", "conclusion": None,
                 "started_at": queued_at, "completed_at": None,
                 "steps": [], "html_url": "https://github.com/o/r/actions/runs/42/job/777"}
            ]
        return 1, None

    out = kc.run_external_ci_watchdog_for_pr(
        conn, task_id, "o/r", 6, head_sha="abc123",
        runner=fake_gh,
        auto_remediation=False,  # no rerun POST during this unit test
        config={"kanban": {"ci_wait": {}}},
    )
    assert out["snapshot"]["head_sha"] == "abc123"
    assert len(out["snapshot"]["jobs"]) >= 1
    # 4000s > stall -> escalated to OWNER_ACTION_REQUIRED (auto off).
    assert out["classification"]["ci_state"] == kc.OWNER_ACTION_REQUIRED
    assert out["emitted"] is True
    # The canonical 10-field alert is persisted on the card.
    comments = kb.list_comments(conn, task_id)
    assert comments
    assert "EXTERNAL-CI-WATCHDOG" in comments[0].body
