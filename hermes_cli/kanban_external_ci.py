"""External-CI wait watchdog for kanban missions.

Hermes missions that deliver a pull request often wait on GitHub Actions
checks. Usually that wait is short and healthy; sometimes the runner queue
stalls and the mission sits on a dependency that will never resolve on its
own. This module classifies that wait from REAL GitHub evidence and drives a
bounded, reversible AUTO remediation — without ever mistaking an external
dependency stall for a Hermes worker stall.

Taxonomy (Product contract "external CI wait watchdog", read-only source of
truth):

* The CI_* states below form an ORTHOGONAL "external dependency status" axis
  of the owning session — they are NEVER a task/worker status. The task stays
  in its normal execution status; the axis explains WHY the mission waits and
  what the operator should see.
* An external CI wait is NEVER classified as a Hermes worker stall: the
  reclaim / heartbeat / recovery worker mechanics do not apply (no worker is
  dead — execution depends on an unavailable GitHub runner).
* Elapsed queue time is the evaluation GATE; GitHub evidence is the
  CLASSIFIER. Neither alone produces a classification. Classification always
  requires a complete, fresh evidence snapshot (see §evidence). Fail-closed
  on missing/incomplete/stale evidence: no stall classification, stay at the
  last valid state (default CI_WAITING) and emit a non-alarmist INFO
  evidence-missing signal.
* ``started_at`` non-null on a queued job is NOT execution proof (queued jobs
  carry started_at = enqueue time). Only ``steps > 0`` or ``status ==
  in_progress`` proves execution.

The operator-visible surface uses ONLY existing tokens (ACTION STATUS
STARTING/RUNNING/RECOVERING/AWAITING_APPROVAL/…, discussion KEEP_OPEN /
OWNER_ACTION_REQUIRED as payload text, attention axis from
``kanban_diagnostics``). The CI_* states stay internal to the watchdog
payload — no invented status token is persisted.

AUTO remediation boundary: authenticated (gh with repo+workflow scopes),
reversible, minimal targeting (job-level rerun first; workflow-level
``rerun-failed-jobs`` fallback only when job-level is unavailable), budget ONE
retry per stall episode, then a bounded observation window. A 403/4xx/5xx API
refusal is NOT recovery — it is the evidence that Hermes cannot retry and the
episode escalates to OWNER_ACTION_REQUIRED (never simulate recovery, never
loop, never cancel/re-push/merge/modify code/fabricate a failure).

Thresholds come from config.yaml ``kanban.ci_wait`` (warning 45 / stall 60 /
retry_window 30 by default, plus optional per-mission overrides keyed by the
umbrella task id). No new ``HERMES_*`` env vars for settings.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# CI state taxonomy (internal classifier values — see module docstring)
# ---------------------------------------------------------------------------

CI_WAITING = "CI_WAITING"                 # checks pending, within normal window
CI_WAITING_LONG = "CI_WAITING_LONG"       # queued beyond warning threshold
CI_INFRA_STALLED = "CI_INFRA_STALLED"     # queued beyond stall threshold, no step
CI_FAILED = "CI_FAILED"                   # execution really started then failed
CI_RECOVERING = "CI_RECOVERING"           # bounded safe retry requested, accepted
# Shared exit/transition labels the classifier can produce (these mirror the
# existing ACTION STATUS / discussion vocabulary, never invented tokens):
RUNNING = "RUNNING"                       # execution started (step > 0)
COMPLETED = "COMPLETED"                   # required checks passed
OWNER_ACTION_REQUIRED = "OWNER_ACTION_REQUIRED"  # Hermes cannot recover bounded

CI_STATES = (
    CI_WAITING,
    CI_WAITING_LONG,
    CI_INFRA_STALLED,
    CI_FAILED,
    CI_RECOVERING,
    RUNNING,
    COMPLETED,
    OWNER_ACTION_REQUIRED,
)

# OWNER ACTION token text (mirrors kanban_diagnostics.OWNER_ACTION_REQUIRED —
# kept local so this module stays dependency-light; no import cycle).
OWNER_ACTION_REQUIRED_TEXT = "REQUIRED"

# Dominance order when several jobs/runs are aggregated: one real failure
# makes the whole set CI_FAILED; otherwise one job past the stall threshold
# stalls it; then warning; then plain waiting.
_DOMINANCE_ORDER = {
    CI_FAILED: 5,
    OWNER_ACTION_REQUIRED: 4,
    CI_INFRA_STALLED: 3,
    CI_WAITING_LONG: 2,
    CI_RECOVERING: 1,
    CI_WAITING: 0,
    RUNNING: 0,
    COMPLETED: -1,
}


# ---------------------------------------------------------------------------
# Config / thresholds (config.yaml kanban.ci_wait)
# ---------------------------------------------------------------------------

DEFAULT_WARNING_MINUTES = 45
DEFAULT_STALL_MINUTES = 60
DEFAULT_RETRY_WINDOW_MINUTES = 30

DEFAULT_CI_WAIT_POLICY: dict = {
    "warning_minutes": DEFAULT_WARNING_MINUTES,
    "stall_minutes": DEFAULT_STALL_MINUTES,
    "retry_window_minutes": DEFAULT_RETRY_WINDOW_MINUTES,
    "mission_overrides": {},
}

_CI_WAIT_MIN = 1
_CI_WAIT_MAX = 24 * 60  # a mission waiting >24h on CI is misconfigured


def _clean_minutes(value: Any, default: int) -> int:
    """Coerce a config minute value to a sane positive int, else ``default``."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if n < _CI_WAIT_MIN or n > _CI_WAIT_MAX:
        return default
    return n


def _clean_mission_overrides(raw: Any) -> dict:
    """Normalise the optional per-mission override map.

    Keys are UMBRELLA TASK IDs (canonical id, never title). A partial override
    (one key set) inherits the other value during resolution.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for mission_id, override in raw.items():
        if not isinstance(override, dict):
            continue
        clean: dict[str, int] = {}
        if "warning_minutes" in override:
            clean["warning_minutes"] = _clean_minutes(
                override.get("warning_minutes"), DEFAULT_WARNING_MINUTES
            )
        if "stall_minutes" in override:
            clean["stall_minutes"] = _clean_minutes(
                override.get("stall_minutes"), DEFAULT_STALL_MINUTES
            )
        if "retry_window_minutes" in override:
            clean["retry_window_minutes"] = _clean_minutes(
                override.get("retry_window_minutes"), DEFAULT_RETRY_WINDOW_MINUTES
            )
        if clean:
            out[str(mission_id)] = clean
    return out


def ci_wait_policy_from_config(config: Optional[dict]) -> dict:
    """Normalise the persisted CI-wait watchdog policy (config.yaml).

    Reads ``kanban.ci_wait``::

        kanban:
          ci_wait:
            warning_minutes: 45        # -> CI_WAITING_LONG
            stall_minutes: 60          # -> CI_INFRA_STALLED
            retry_window_minutes: 30   # bounded post-retry observation window
            mission_overrides:         # optional; key = umbrella task id
              t_xxxxxxxx:
                warning_minutes: 60
                stall_minutes: 120

    Precedence: ``mission_overrides[umbrella_id]`` > ``kanban.ci_wait.*``
    defaults > built-in 45/60/30. Configuration is advisory, never
    load-bearing for safety: invalid values (<=0, wrong types, out of range)
    fall back to the defaults with no error, mirroring
    ``autonomy_policy_from_config``.

    Coherence: if the resolved ``warning_minutes >= stall_minutes`` the
    CI_WAITING_LONG state is never emitted — the transition goes straight
    from CI_WAITING to CI_INFRA_STALLED at the stall threshold.
    """
    policy = dict(DEFAULT_CI_WAIT_POLICY)
    kanban = config.get("kanban") if isinstance(config, dict) else None
    raw = kanban.get("ci_wait") if isinstance(kanban, dict) else None
    if not isinstance(raw, dict):
        return policy
    policy["warning_minutes"] = _clean_minutes(
        raw.get("warning_minutes"), DEFAULT_WARNING_MINUTES
    )
    policy["stall_minutes"] = _clean_minutes(
        raw.get("stall_minutes"), DEFAULT_STALL_MINUTES
    )
    policy["retry_window_minutes"] = _clean_minutes(
        raw.get("retry_window_minutes"), DEFAULT_RETRY_WINDOW_MINUTES
    )
    policy["mission_overrides"] = _clean_mission_overrides(raw.get("mission_overrides"))
    return policy


def resolve_ci_wait_policy(
    policy: Optional[dict], umbrella_id: Optional[str] = None
) -> dict:
    """Resolve the effective thresholds for a mission (override > defaults).

    ``policy`` is the output of :func:`ci_wait_policy_from_config`. A partial
    mission override inherits the unset values from the global defaults.
    """
    base = dict(DEFAULT_CI_WAIT_POLICY, **(policy or {}))
    effective = {
        "warning_minutes": _clean_minutes(
            base.get("warning_minutes"), DEFAULT_WARNING_MINUTES
        ),
        "stall_minutes": _clean_minutes(
            base.get("stall_minutes"), DEFAULT_STALL_MINUTES
        ),
        "retry_window_minutes": _clean_minutes(
            base.get("retry_window_minutes"), DEFAULT_RETRY_WINDOW_MINUTES
        ),
    }
    if umbrella_id:
        overrides = base.get("mission_overrides") or {}
        override = overrides.get(str(umbrella_id))
        if isinstance(override, dict):
            if "warning_minutes" in override:
                effective["warning_minutes"] = _clean_minutes(
                    override.get("warning_minutes"),
                    effective["warning_minutes"],
                )
            if "stall_minutes" in override:
                effective["stall_minutes"] = _clean_minutes(
                    override.get("stall_minutes"), effective["stall_minutes"]
                )
            if "retry_window_minutes" in override:
                effective["retry_window_minutes"] = _clean_minutes(
                    override.get("retry_window_minutes"),
                    effective["retry_window_minutes"],
                )
    effective["warning_enabled"] = (
        effective["warning_minutes"] < effective["stall_minutes"]
    )
    return effective


# ---------------------------------------------------------------------------
# Evidence snapshot
# ---------------------------------------------------------------------------

# Required evidence keys for a stall-capable classification (Product §2).
_EVIDENCE_REQUIRED_KEYS = (
    "captured_at",      # int epoch when the snapshot was fetched
    "repo",             # "owner/name"
    "pr_number",        # int
    "head_sha",         # current PR head sha
    "runs",             # list[dict] — workflow runs for the PR
    "jobs",             # list[dict] — jobs of the evaluated run(s)
)


@dataclass
class EvidenceSnapshot:
    """A validated, timestamped GitHub evidence snapshot (no DB writes).

    All fields are raw REST-shaped values collected by
    :func:`collect_external_ci_snapshot` (or hand-built in tests). The
    classifier treats missing/incoherent fields as absent evidence — it never
    guesses.
    """

    captured_at: int = 0
    repo: str = ""
    pr_number: Optional[int] = None
    head_sha: str = ""
    runs: list[dict] = field(default_factory=list)
    jobs: list[dict] = field(default_factory=list)
    # True when the observed workflow/check is still REQUIRED for the merge
    # (PR open, check not obsolete/superseded). Absent -> not required.
    required: bool = False
    # A newer workflow run (same workflow+branch/PR, or a newer SHA) exists.
    superseded: bool = False
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "EvidenceSnapshot":
        data = data or {}
        return cls(
            captured_at=int(data.get("captured_at") or 0),
            repo=str(data.get("repo") or ""),
            pr_number=data.get("pr_number"),
            head_sha=str(data.get("head_sha") or ""),
            runs=list(data.get("runs") or []),
            jobs=list(data.get("jobs") or []),
            required=bool(data.get("required")),
            superseded=bool(data.get("superseded")),
            raw=data,
        )

    def to_dict(self) -> dict:
        return {
            "captured_at": self.captured_at,
            "repo": self.repo,
            "pr_number": self.pr_number,
            "head_sha": self.head_sha,
            "runs": self.runs,
            "jobs": self.jobs,
            "required": self.required,
            "superseded": self.superseded,
        }

    def evidence_complete(self) -> bool:
        """Fail-closed gate: is the snapshot complete enough to classify?

        A stall classification requires every evidence field: a fresh
        timestamp, a named repo/PR/SHA, at least one run, and jobs for that
        run with steps visible. Missing any of these -> incomplete.
        """
        if not self.captured_at or not self.repo:
            return False
        if not self.pr_number or not self.head_sha:
            return False
        if not self.runs:
            return False
        if not self.jobs:
            return False
        # At least one job row must carry the fields the classifier reads.
        for job in self.jobs:
            if not isinstance(job, dict):
                continue
            if "status" in job and "id" in job:
                return True
        return False


# ---------------------------------------------------------------------------
# Evidence helpers (pure)
# ---------------------------------------------------------------------------


def _parse_github_ts(value: Any) -> Optional[int]:
    """Parse a GitHub REST ISO-8601 timestamp (e.g. 2026-09-05T04:28:00Z)
    into an epoch int. Returns None on empty/malformed input."""
    if not value:
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


def _job_status(job: dict) -> str:
    return str((job or {}).get("status") or "")


def _job_conclusion(job: dict) -> Optional[str]:
    value = (job or {}).get("conclusion")
    return str(value) if value else None


def _job_steps(job: dict) -> list:
    steps = (job or {}).get("steps")
    return list(steps) if isinstance(steps, list) else []


def _job_started_at(job: dict) -> Optional[int]:
    value = (job or {}).get("started_at")
    return _parse_github_ts(value)


def job_execution_started(job: dict) -> bool:
    """True only when execution is PROVEN (historical OR current).

    ``status == in_progress`` or at least one step row exists. A non-null
    ``started_at`` on a queued job is NOT proof (queued jobs carry
    started_at = enqueue time) — Product finding PR #6.
    """
    if _job_status(job) == "in_progress":
        return True
    return bool(_job_steps(job))


def job_is_running(job: dict) -> bool:
    """True when the job is executing RIGHT NOW (not yet terminal).

    A completed job carries historical steps but is no longer running — it
    must not keep the set in RUNNING once every check has finished. Only
    ``in_progress`` (or a queued/waiting job that already has steps, an
    inconsistent-but-observable state) counts as current execution.
    """
    status = _job_status(job)
    if status == "completed":
        return False
    if status == "in_progress":
        return True
    return bool(_job_steps(job))


def job_is_failed(job: dict) -> bool:
    """Real completed failure: execution started AND conclusion == failure."""
    if _job_conclusion(job) != "failure":
        return False
    # A job that reached a failure conclusion necessarily ran; still require
    # execution proof so a malformed payload never fabricates a failure.
    return job_execution_started(job) or _job_status(job) == "completed"


def job_is_queued(job: dict) -> bool:
    return _job_status(job) == "queued" and _job_conclusion(job) is None


def run_has_failure_conclusion(run: dict) -> bool:
    return str((run or {}).get("conclusion") or "") == "failure"


def _run_created_at(run: dict) -> Optional[int]:
    return _parse_github_ts((run or {}).get("created_at"))


def _max_queue_minutes(snapshot: EvidenceSnapshot, now: int) -> int:
    """Longest observed queue wait among required queued jobs.

    Queue duration measures from the job's queued_at / run created_at up to
    the snapshot captured_at (never from wall-clock alone).
    """
    captured = snapshot.captured_at or now
    run_by_id = {str((r or {}).get("id")): r for r in snapshot.runs}
    longest = 0
    for job in snapshot.jobs:
        if not job_is_queued(job):
            continue
        run = run_by_id.get(str((job or {}).get("run_id")))
        queued_at = _job_started_at(job)  # queued jobs: started_at == queue time
        if not queued_at and run is not None:
            queued_at = _run_created_at(run)
        if not queued_at:
            continue
        minutes = max(0, int((captured - int(queued_at)) / 60))
        longest = max(longest, minutes)
    return longest


def aggregate_ci_states(states: list[str]) -> str:
    """Apply the multi-job/multi-run dominance rule.

    CI_FAILED > OWNER_ACTION_REQUIRED > CI_INFRA_STALLED > CI_WAITING_LONG >
    CI_RECOVERING > CI_WAITING/RUNNING > COMPLETED. A single real failure
    makes the whole set CI_FAILED; otherwise a single job past a threshold
    classifies the set at that level. Healthy terminal states never dominate
    an alert.
    """
    if not states:
        return CI_WAITING
    ranked = [s for s in states if s in _DOMINANCE_ORDER]
    if not ranked:
        return CI_WAITING
    ranked.sort(key=lambda s: _DOMINANCE_ORDER[s], reverse=True)
    return ranked[0]


# ---------------------------------------------------------------------------
# Classifier (pure — the heart of the watchdog)
# ---------------------------------------------------------------------------


# Freshness bound for a classification-capable snapshot. Elapsed queue time
# is measured against the snapshot's own captured_at; a snapshot OLDER than
# this bound is stale evidence -> fail-closed (no stall classification).
# Default 2x a 5-minute evaluation tick (Product §2: freshness <= 2x tick).
DEFAULT_SNAPSHOT_MAX_AGE_SECONDS = 2 * 300


def _snapshot_queued_job_ids(snapshot: EvidenceSnapshot) -> list[str]:
    return [
        str((j or {}).get("id"))
        for j in snapshot.jobs
        if isinstance(j, dict) and job_is_queued(j)
    ]


def _snapshot_failed_job_ids(snapshot: EvidenceSnapshot) -> list[str]:
    return [
        str((j or {}).get("id"))
        for j in snapshot.jobs
        if isinstance(j, dict) and job_is_failed(j)
    ]


def _current_run_id(snapshot: EvidenceSnapshot) -> Optional[str]:
    """Id of the newest observed run (the one being evaluated)."""
    runs = [r for r in snapshot.runs if isinstance(r, dict) and r.get("id")]
    if not runs:
        return None

    def _key(run: dict) -> tuple[int, int]:
        return (
            int(run.get("id") or 0),
            int(_parse_github_ts(run.get("created_at")) or 0),
        )

    runs.sort(key=_key, reverse=True)
    return str(runs[0].get("id"))


def classify_external_ci_wait(
    snapshot: EvidenceSnapshot,
    *,
    policy: Optional[dict] = None,
    now: Optional[int] = None,
    snapshot_max_age_seconds: Optional[int] = DEFAULT_SNAPSHOT_MAX_AGE_SECONDS,
) -> dict:
    """Classify an external-CI wait from evidence, never elapsed time alone.

    Returns a structured classification dict::

        {
          "ci_state": CI_WAITING | CI_WAITING_LONG | CI_INFRA_STALLED |
                      CI_FAILED | CI_RECOVERING | RUNNING | COMPLETED |
                      OWNER_ACTION_REQUIRED,
          "evidence_ok": bool,      # snapshot complete AND fresh enough
          "superseded": bool,       # a newer run supersedes the observed one
          "queue_minutes": int,     # longest observed required queue wait
          "execution_started": bool,
          "failedJobIds": [...],    # evidence: completed-failure job ids
          "queuedJobIds": [...],    # evidence: queued job ids
          "runId": str|None,
          "warning_minutes": int,
          "stall_minutes": int,
          "reason": str,            # human-readable EN summary
        }

    Fail-closed on evidence: an incomplete/absent/stale snapshot never
    classifies a stall — the state stays CI_WAITING with ``evidence_ok=False``
    and a non-alarmist INFO signal (caller surfaces it as evidence-missing,
    never as CRITICAL without proof).
    """
    effective = resolve_ci_wait_policy(policy)
    now_i = int(now if now is not None else time.time())
    warning = effective["warning_minutes"]
    stall = effective["stall_minutes"]
    warning_enabled = effective["warning_enabled"]
    max_age = snapshot_max_age_seconds

    stale = bool(
        snapshot.captured_at
        and max_age is not None
        and (now_i - int(snapshot.captured_at)) > int(max_age)
    )

    if not snapshot.evidence_complete() or stale:
        return {
            "ci_state": CI_WAITING,
            "evidence_ok": False,
            "superseded": bool(snapshot.superseded),
            "queue_minutes": 0,
            "execution_started": False,
            "failedJobIds": [],
            "queuedJobIds": [],
            "runId": None,
            "warning_minutes": warning,
            "stall_minutes": stall,
            "reason": (
                "Evidence snapshot incomplete or stale — no CI "
                "classification emitted (fail-closed)."
            ),
        }

    # Supersession (anti-false-alert): evaluation only on the CURRENT run.
    # A queued run on a superseded SHA never alerts.
    if snapshot.superseded:
        return {
            "ci_state": CI_WAITING,
            "evidence_ok": True,
            "superseded": True,
            "queue_minutes": _max_queue_minutes(snapshot, now_i),
            "execution_started": False,
            "failedJobIds": [],
            "queuedJobIds": _snapshot_queued_job_ids(snapshot),
            "runId": _current_run_id(snapshot),
            "warning_minutes": warning,
            "stall_minutes": stall,
            "reason": "Workflow run superseded by a newer run/SHA — no alert "
                      "for the obsolete queued run.",
        }

    # The workflow/check is no longer REQUIRED (PR closed / merged, check
    # obsolete): the wait is moot — silent exit, never an alert.
    if not snapshot.required:
        return {
            "ci_state": COMPLETED,
            "evidence_ok": True,
            "superseded": False,
            "queue_minutes": _max_queue_minutes(snapshot, now_i),
            "execution_started": False,
            "failedJobIds": [],
            "queuedJobIds": [],
            "runId": _current_run_id(snapshot),
            "warning_minutes": warning,
            "stall_minutes": stall,
            "reason": "The workflow/check is no longer required (PR closed or "
                      "obsolete) — no CI wait alert.",
        }

    # CI_FAILED requires real execution proof then failure.
    failed_jobs = _snapshot_failed_job_ids(snapshot)
    run_failed = any(
        isinstance(r, dict) and run_has_failure_conclusion(r)
        for r in snapshot.runs
    )
    if failed_jobs or run_failed:
        return {
            "ci_state": CI_FAILED,
            "evidence_ok": True,
            "superseded": False,
            "queue_minutes": _max_queue_minutes(snapshot, now_i),
            "execution_started": True,
            "failedJobIds": failed_jobs,
            "queuedJobIds": _snapshot_queued_job_ids(snapshot),
            "runId": _current_run_id(snapshot),
            "warning_minutes": warning,
            "stall_minutes": stall,
            "reason": "Execution really started then failed — this is a code/"
                      "test failure, NOT an infrastructure wait.",
        }

    # Execution started (even if the overall queue is long) -> natural RUNNING.
    # A COMPLETED job (historical steps) is NOT running — terminal jobs fall
    # through to the COMPLETED/queue logic below.
    if any(
        isinstance(j, dict) and job_is_running(j) for j in snapshot.jobs
    ):
        return {
            "ci_state": RUNNING,
            "evidence_ok": True,
            "superseded": False,
            "queue_minutes": _max_queue_minutes(snapshot, now_i),
            "execution_started": True,
            "failedJobIds": [],
            "queuedJobIds": _snapshot_queued_job_ids(snapshot),
            "runId": _current_run_id(snapshot),
            "warning_minutes": warning,
            "stall_minutes": stall,
            "reason": "A job is executing right now — the CI is running normally.",
        }

    # No required job queued and nothing running -> checks resolved.
    queued_ids = _snapshot_queued_job_ids(snapshot)
    if not queued_ids:
        return {
            "ci_state": COMPLETED,
            "evidence_ok": True,
            "superseded": False,
            "queue_minutes": 0,
            "execution_started": False,
            "failedJobIds": [],
            "queuedJobIds": [],
            "runId": _current_run_id(snapshot),
            "warning_minutes": warning,
            "stall_minutes": stall,
            "reason": "No required check is queued and no failure is observed "
                      "— required checks have resolved.",
        }

    # Elapsed queue time is the GATE; the snapshot above is the classifier.
    queue_minutes = _max_queue_minutes(snapshot, now_i)
    if queue_minutes >= stall:
        state = CI_INFRA_STALLED
        reason = (
            f"Required job(s) queued {queue_minutes}m (>= stall {stall}m) with "
            "no step started and no completed failure — infrastructure stall."
        )
    elif warning_enabled and queue_minutes >= warning:
        state = CI_WAITING_LONG
        reason = (
            f"Required job(s) queued {queue_minutes}m (>= warning {warning}m) "
            "with no step started — prolonged external wait."
        )
    else:
        state = CI_WAITING
        reason = (
            f"Required job(s) queued {queue_minutes}m — within the normal "
            "window; no alert."
        )
    return {
        "ci_state": state,
        "evidence_ok": True,
        "superseded": False,
        "queue_minutes": queue_minutes,
        "execution_started": False,
        "failedJobIds": [],
        "queuedJobIds": queued_ids,
        "runId": _current_run_id(snapshot),
        "warning_minutes": warning,
        "stall_minutes": stall,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Operator alert builder (10-field schema, UX copy)
# ---------------------------------------------------------------------------

_ATTENTION_BY_STATE = {
    CI_WAITING: "NONE",
    CI_WAITING_LONG: "WARNING",
    CI_INFRA_STALLED: "ACTION_REQUIRED",
    CI_FAILED: "ACTION_REQUIRED",
    CI_RECOVERING: "WARNING",
    RUNNING: "INFO",
    COMPLETED: "INFO",
    OWNER_ACTION_REQUIRED: "ACTION_REQUIRED",
}

# discussion.status text carried in the payload (the lifecycle tokens
# KEEP_OPEN / OWNER_ACTION_REQUIRED are the existing discussion vocabulary;
# when the discussion-lifecycle schema is not merged on the fork the payload
# still carries the value as operator copy — no invented status is written).
_DISCUSSION_BY_STATE = {
    CI_WAITING: "KEEP_OPEN",
    CI_WAITING_LONG: "KEEP_OPEN",
    CI_INFRA_STALLED: "KEEP_OPEN",   # bounded AUTO retry still possible
    CI_FAILED: "OWNER_ACTION_REQUIRED",
    CI_RECOVERING: "KEEP_OPEN",
    RUNNING: "KEEP_OPEN",
    COMPLETED: "KEEP_OPEN",
    OWNER_ACTION_REQUIRED: "OWNER_ACTION_REQUIRED",
}

_ACTION_STATUS_BY_STATE = {
    CI_WAITING: "RUNNING",                # unchanged, still legitimately waiting
    CI_WAITING_LONG: "RUNNING",           # unchanged — KEEP_OPEN
    CI_INFRA_STALLED: "RUNNING",          # unchanged at block time (KEEP_OPEN)
    CI_FAILED: "FAILED",
    CI_RECOVERING: "RECOVERING",
    RUNNING: "RUNNING",
    COMPLETED: "COMPLETED",
    OWNER_ACTION_REQUIRED: "AWAITING_APPROVAL",
}


def _owner_action_for_ci(state: str, auto_remediation_available: bool) -> str:
    if state in (CI_FAILED, OWNER_ACTION_REQUIRED):
        return "REQUIRED"
    if state == CI_INFRA_STALLED and not auto_remediation_available:
        return "REQUIRED"
    return "NONE"


def _dur_fr(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}"


# Per-state operator copy (10-field schema, EN labels, FR copy — UX contract
# §2). ``owner`` is the OWNER ACTION token; ``required``/``why`` are added
# under OWNER ACTION when REQUIRED.
_STATE_COPY: dict[str, dict] = {
    CI_WAITING: {
        "mission_status": (
            "En attente de la CI externe — la mission reste ouverte, aucune "
            "action requise."
        ),
        "impact": "Aucun — attente dans la fenêtre normale.",
        "recommended": "Aucune.",
        "next": "Watchdog — alerte si l'attente dépasse le seuil.",
        "discussion_action": "Aucune.",
    },
    CI_WAITING_LONG: {
        "mission_status": (
            "En attente de la CI externe — la mission reste ouverte, aucune "
            "action requise."
        ),
        "impact": (
            "Le delivery reste en attente ; la mission ne progresse pas mais "
            "aucun échec n'est observé."
        ),
        "recommended": (
            "Aucune pour l'instant — continuer à surveiller ; le retry "
            "automatique n'interviendra qu'au-delà du seuil de blocage."
        ),
        "next": (
            "Watchdog — reclasser CI_INFRA_STALLED si l'attente dépasse le "
            "seuil de blocage ; RUNNING dès qu'un step démarre."
        ),
        "discussion_action": "Aucune — session maintenue ouverte (KEEP_OPEN).",
    },
    CI_INFRA_STALLED: {
        "mission_status": (
            "Dépendance externe bloquée — la mission reste ouverte ; ce n'est "
            "PAS un stall worker Hermes (external dependency stall)."
        ),
        "impact": "Le delivery de {pr} reste bloqué.",
        "recommended": (
            "Relancer uniquement les jobs queued concernés (job-level rerun)."
        ),
        "next": (
            "CI_RECOVERING (retry émis) → RUNNING (un step démarre) → "
            "COMPLETED (checks passés) ; OWNER_ACTION_REQUIRED si le retry "
            "borné reste queued."
        ),
        "discussion_action": "Aucune (retry borné en cours).",
    },
    CI_FAILED: {
        "mission_status": (
            "Échec d'exécution CI — distinct d'une attente d'infrastructure."
        ),
        "impact": (
            "La PR ne peut pas être livrée tant que l'échec n'est pas traité."
        ),
        "recommended": (
            "Analyser l'échec puis lancer la remediation ciblée (correction + "
            "re-run du job concerné) — jamais un rerun global non nécessaire."
        ),
        "next": "Remediation ciblée → re-run → RUNNING → COMPLETED.",
        "discussion_action": "Voir le job en échec / approuver la remediation ciblée.",
    },
    CI_RECOVERING: {
        "mission_status": (
            "Retry borné en cours — la mission reste ouverte (KEEP_OPEN)."
        ),
        "impact": (
            "Attente prolongée possible ; la mission reste bloquée sur la "
            "dépendance externe tant que le retry ne démarre pas."
        ),
        "recommended": "Aucune — fenêtre bornée en cours.",
        "next": (
            "RUNNING dès qu'un step démarre ; COMPLETED si les checks passent ; "
            "OWNER_ACTION_REQUIRED si le retry reste queued au-delà de la "
            "fenêtre."
        ),
        "discussion_action": "Aucune.",
    },
    OWNER_ACTION_REQUIRED: {
        "mission_status": (
            "Hermes ne peut pas relancer automatiquement — décision opérateur "
            "requise."
        ),
        "impact": (
            "Le delivery de {pr} reste bloqué tant que les jobs ne sont pas "
            "relancés."
        ),
        "recommended": "Relancer uniquement les jobs queued concernés.",
        "next": (
            "Après rerun → RUNNING (un step démarre) → COMPLETED (checks "
            "passés)."
        ),
        "discussion_action": "Relancer uniquement les jobs queued concernés ({pr}).",
    },
    RUNNING: {
        "mission_status": "CI en exécution — la mission continue.",
        "impact": "Aucun — la CI progresse.",
        "recommended": "Aucune.",
        "next": "COMPLETED si tous les checks passent ; CI_FAILED si un job échoue.",
        "discussion_action": "Aucune.",
    },
    COMPLETED: {
        "mission_status": "Dépendance CI résolue — checks passés.",
        "impact": "Aucun — le delivery peut reprendre.",
        "recommended": "Poursuivre le workflow de la mission.",
        "next": "Suite normale de la mission (delivery / gate suivant).",
        "discussion_action": "Aucune.",
    },
}

_CI_FAILED_REQUIRED_ACTION = (
    "Analyser l'échec réel puis lancer la remediation ciblée ({pr}) — "
    "correction + re-run du job concerné."
)
_CI_FAILED_WHY = (
    "L'infrastructure a exécuté le job puis il a échoué — ce n'est pas une "
    "attente de runner ; le job-level rerun ciblé est l'action la plus sûre."
)
_STALL_REQUIRED_ACTION = (
    "Relancer uniquement les jobs queued concernés de la {pr} (job-level rerun)."
)
_STALL_WHY = (
    "L'attente dépasse la fenêtre de retry automatique bornée ; aucun échec "
    "de code/test n'est observé — le job-level rerun des seuls jobs concernés "
    "est l'action la plus sûre (pas de rerun global, pas de modification de "
    "code, pas de merge)."
)


def build_operator_alert(
    classification: dict,
    *,
    mission_title: str = "",
    pr_url: str = "",
    retry_available: bool = False,
    retry_evidence: Optional[dict] = None,
) -> dict:
    """Build the 10-field operator alert payload (canonical order, EN labels,
    FR operator copy). The CI_* state stays internal — the payload carries the
    existing attention / ACTION STATUS / discussion vocabulary.

    Returns a dict with keys in the canonical handoff order:
    MISSION STATUS / SUMMARY / EXTERNAL DEPENDENCY STATUS / CI EVIDENCE /
    IMPACT / RECOMMENDED ACTION / OWNER ACTION / NEXT ACTION /
    DISCUSSION STATUS / DISCUSSION ACTION (plus REQUIRED ACTION + WHY under
    OWNER ACTION when REQUIRED, and the internal ``ciState`` for the board).
    """
    state = classification.get("ci_state") or CI_WAITING
    attention = _ATTENTION_BY_STATE.get(state, "INFO")
    owner_action = _owner_action_for_ci(state, retry_available)
    queue_minutes = int(classification.get("queue_minutes") or 0)
    evidence_ok = bool(classification.get("evidence_ok"))
    repo = classification.get("repo") or ""
    pr = classification.get("pr_number")
    pr_ref = f"PR #{pr}" if pr else (pr_url or "PR")
    copy = _STATE_COPY.get(state, _STATE_COPY[CI_WAITING])

    # --- summary head line (per-state, factual) ---
    summary = classification.get("reason") or ""
    if state == CI_WAITING_LONG:
        summary = (
            f"GitHub Actions attend un runner depuis {_dur_fr(queue_minutes)}. "
            "Aucun test n'a encore démarré."
        )
    elif state == CI_INFRA_STALLED:
        summary = (
            f"GitHub Actions attend un runner depuis {_dur_fr(queue_minutes)}. "
            "Aucun test n'a encore démarré. Aucun échec de code n'est observé."
        )
    elif state == CI_FAILED:
        summary = (
            "Un test a réellement démarré puis échoué. Ce n'est pas une "
            "attente d'infrastructure."
        )
    elif state == CI_RECOVERING:
        summary = (
            "Hermes a relancé uniquement les jobs queued concernés. GitHub "
            "Actions n'a toujours pas démarré de test."
        )
    elif state == OWNER_ACTION_REQUIRED:
        summary = (
            "Les jobs restent queued au-delà de la fenêtre de retry "
            "automatique bornée (ou le retry est indisponible). Aucun test "
            "n'a démarré, aucun échec de code n'est observé."
        )
    elif state == RUNNING:
        summary = "Un test a démarré sur GitHub Actions. La CI progresse normalement."
    elif state == COMPLETED:
        summary = (
            "Tous les checks requis sont passés. La mission peut poursuivre "
            "son workflow."
        )
    elif state == CI_WAITING and not evidence_ok:
        summary = (
            "Impossible de vérifier l'état GitHub — snapshot indisponible ou "
            "incomplet ; aucune classification de blocage émise (fail-closed)."
        )
    elif state == CI_WAITING:
        summary = (
            f"GitHub Actions attend un runner depuis {_dur_fr(queue_minutes)}. "
            "Aucune alerte — attente dans la fenêtre normale."
        )

    # CI EVIDENCE: real snapshot fields, never elapsed time alone.
    if evidence_ok:
        queued_txt = ", ".join(
            str(j) for j in (classification.get("queuedJobIds") or [])[:5]
        ) or "aucun job queued"
        failed_txt = ", ".join(
            str(j) for j in (classification.get("failedJobIds") or [])[:5]
        )
        superseded_txt = (
            " ; workflow run superseding présent — aucune alerte sur l'ancien run"
            if classification.get("superseded")
            else " ; aucun workflow run superseding"
        )
        ci_evidence = (
            f"run {classification.get('runId') or '?'} — jobs {queued_txt} "
            f"queued, steps=0 ; aucune exécution démarrée{superseded_txt}"
            + (f" ; échec(s) complété(s): {failed_txt}" if failed_txt else "")
        )
    else:
        ci_evidence = (
            "Impossible de vérifier l'état GitHub — snapshot indisponible ou "
            "incomplet (fail-closed : aucune classification de blocage émise)."
        )

    def _fmt(template: str) -> str:
        return template.format(pr=pr_ref)

    alert: dict = {
        # Canonical 10-field schema (in order).
        "missionStatus": _fmt(copy["mission_status"]),
        "summary": summary,
        "externalDependencyStatus": state,
        "ciEvidence": ci_evidence,
        "impact": _fmt(copy["impact"]),
        "recommendedAction": _fmt(copy["recommended"]),
        "ownerAction": owner_action,
        "nextAction": _fmt(copy["next"]),
        "discussionStatus": _DISCUSSION_BY_STATE.get(state, "KEEP_OPEN"),
        "discussionAction": _fmt(copy["discussion_action"]),
        # Operator axis / internal classifier values (existing tokens only).
        "attention": attention,
        "actionStatus": _ACTION_STATUS_BY_STATE.get(state, "RUNNING"),
        "ciState": state,
        "queueMinutes": queue_minutes,
        "evidenceOk": evidence_ok,
        "superseded": bool(classification.get("superseded")),
        "missionTitle": mission_title,
        "prUrl": pr_url,
        "retryAvailable": retry_available,
        "retryEvidence": retry_evidence,
        "capturedAt": classification.get("captured_at"),
        "reason": classification.get("reason"),
    }
    if owner_action == OWNER_ACTION_REQUIRED_TEXT:
        if state == CI_FAILED:
            alert["requiredAction"] = _CI_FAILED_REQUIRED_ACTION.format(pr=pr_ref)
            alert["why"] = _CI_FAILED_WHY
        else:
            alert["requiredAction"] = _STALL_REQUIRED_ACTION.format(pr=pr_ref)
            alert["why"] = _STALL_WHY
    return alert


def render_alert_text(alert: dict) -> str:
    """Render the operator alert as a card comment (10-field schema in order)."""
    lines = [
        f"EXTERNAL DEPENDENCY STATUS : {alert.get('externalDependencyStatus')}",
        f"MISSION STATUS : {alert.get('missionStatus')}",
        f"SUMMARY : {alert.get('summary')}",
        f"CI EVIDENCE : {alert.get('ciEvidence')}",
        f"IMPACT : {alert.get('impact')}",
        f"RECOMMENDED ACTION : {alert.get('recommendedAction')}",
        f"OWNER ACTION : {alert.get('ownerAction')}",
    ]
    if alert.get("ownerAction") == OWNER_ACTION_REQUIRED_TEXT:
        lines.append(f"REQUIRED ACTION : {alert.get('requiredAction')}")
        lines.append(f"WHY : {alert.get('why')}")
    lines.extend(
        [
            f"NEXT ACTION : {alert.get('nextAction')}",
            f"DISCUSSION STATUS : {alert.get('discussionStatus')}",
            f"DISCUSSION ACTION : {alert.get('discussionAction')}",
            f"CI STATE (interne) : {alert.get('ciState')}",
            f"ATTENTION : {alert.get('attention')}",
            f"ACTION STATUS : {alert.get('actionStatus')}",
        ]
    )
    return "\n".join(str(line) for line in lines if line is not None)


# ---------------------------------------------------------------------------
# GitHub evidence collection (gh CLI, REST, read-only)
# ---------------------------------------------------------------------------

GH_TIMEOUT_SECONDS = 30


def gh_available() -> bool:
    """Return whether the gh CLI is on PATH (authentication checked by caller)."""
    return shutil.which("gh") is not None


#: A ``gh api`` runner returns ``(status_code, body)`` where body is raw text
#: or a parsed JSON value (dict/list/None).
GHRunner = Optional[Callable[[list[str]], "tuple[int, Any]"]]


def _gh_json(
    args: list[str],
    *,
    runner: GHRunner = None,
) -> tuple[int, Any]:
    """Run ``gh api ...`` and parse JSON. Returns (status_code, parsed).

    ``runner`` is injectable for tests; default runs the real gh CLI with a
    bounded timeout and never logs the token.
    """
    if runner is not None:
        return runner(args)
    if not gh_available():
        return 1, None
    try:
        proc = subprocess.run(
            ["gh", "api", *args],
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return 1, None
    except Exception:
        return 1, None
    body = (proc.stdout or "").strip()
    if proc.returncode != 0:
        # gh writes errors to stderr and may still emit a JSON body.
        if body:
            try:
                return proc.returncode, json.loads(body)
            except Exception:
                return proc.returncode, None
        return proc.returncode, None
    if not body:
        return proc.returncode, {}
    try:
        return proc.returncode, json.loads(body)
    except Exception:
        return proc.returncode, None


def collect_external_ci_snapshot(
    repo: str,
    pr_number: int,
    *,
    head_sha: Optional[str] = None,
    runner: GHRunner = None,
    captured_at: Optional[int] = None,
) -> EvidenceSnapshot:
    """Collect a fresh GitHub evidence snapshot for a PR (read-only REST).

    Fetches: the PR (head sha, mergeability), the workflow runs for that head
    sha, and the jobs of the most recent run. Any REST failure yields an
    incomplete snapshot — the classifier then fails closed (never guesses).

    ``runner`` is injectable for tests. When omitted, the real ``gh api`` CLI
    is used (requires authenticated gh with repo+workflow scopes).
    """
    now_i = int(captured_at if captured_at is not None else time.time())
    raw: dict = {"captured_at": now_i, "repo": repo, "pr_number": pr_number}

    status, pr = _gh_json(
        [f"repos/{repo}/pulls/{pr_number}"], runner=runner
    )
    if status != 0 or not isinstance(pr, dict):
        return EvidenceSnapshot.from_dict(raw)
    head_sha = head_sha or str(pr.get("head", {}).get("sha") or "")
    raw["head_sha"] = head_sha
    raw["pr_open"] = str(pr.get("state") or "") == "open"
    # The check is "required" while the PR is open (still mergeable-targeted).
    raw["required"] = raw["pr_open"]

    status, runs = _gh_json(
        [
            f"repos/{repo}/actions/runs?head_sha={head_sha}&per_page=100",
            "--jq", ".workflow_runs",
        ],
        runner=runner,
    )
    if status != 0 or not isinstance(runs, list):
        return EvidenceSnapshot.from_dict(raw)
    raw["runs"] = runs

    # Order runs newest-first, take the newest (pull_request event preferred).
    def _run_key(run: dict) -> tuple[int, int]:
        return (
            int((run or {}).get("id") or 0),
            int(_parse_github_ts((run or {}).get("created_at")) or 0),
        )

    ordered = sorted(runs, key=_run_key, reverse=True)
    current: Optional[dict] = None
    for run in ordered:
        if str((run or {}).get("head_sha") or "") == head_sha:
            current = run
            break
    if current is None and ordered:
        current = ordered[0]

    if current is None:
        return EvidenceSnapshot.from_dict(raw)

    run_id = (current or {}).get("id")
    raw["run"] = current
    raw["superseded"] = False
    # A run is superseded when a NEWER run exists for the same head_sha that
    # replaces the observed one (higher id = started later for same SHA, or a
    # later pull_request event run).
    for run in ordered:
        if run is current:
            break
        if str((run or {}).get("head_sha") or "") == head_sha:
            raw["superseded"] = True
            break

    status, jobs = _gh_json(
        [
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
            "--jq", ".jobs",
        ],
        runner=runner,
    )
    if status != 0 or not isinstance(jobs, list):
        return EvidenceSnapshot.from_dict(raw)
    # Attach step-level facts (steps present + started) — the execution proof.
    normalized_jobs: list[dict] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        steps = job.get("steps") or []
        normalized_jobs.append(
            {
                "id": job.get("id"),
                "run_id": job.get("run_id"),
                "name": job.get("name"),
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "steps": steps,
                "html_url": job.get("html_url"),
            }
        )
    raw["jobs"] = normalized_jobs
    return EvidenceSnapshot.from_dict(raw)


# ---------------------------------------------------------------------------
# Bounded AUTO remediation (reversible, minimal, authenticated)
# ---------------------------------------------------------------------------

RETRY_JOB_LEVEL = "job_level"
RETRY_WORKFLOW_LEVEL = "workflow_failed_jobs"
RETRY_NONE = "none"

# Max number of retry attempts per stall episode (budget: exactly ONE).
RETRY_BUDGET_PER_EPISODE = 1


@dataclass
class RetryResult:
    """Outcome of a bounded AUTO retry attempt.

    ``accepted`` is True only when the GitHub API returned 2xx. A 4xx/5xx
    refusal is ``accepted=False`` with the refusal evidence — the caller
    escalates to OWNER_ACTION_REQUIRED, never simulates recovery.
    """

    accepted: bool = False
    level: str = RETRY_NONE
    status_code: Optional[int] = None
    body: Any = None
    job_id: Optional[str] = None
    run_id: Optional[str] = None
    attempted_at: int = 0

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "level": self.level,
            "statusCode": self.status_code,
            "body": self.body,
            "jobId": self.job_id,
            "runId": self.run_id,
            "attemptedAt": self.attempted_at,
        }


def _gh_post(
    args: list[str],
    *,
    runner: GHRunner = None,
) -> tuple[int, Any]:
    """POST via gh api. Returns (status_code, parsed_body_or_text)."""
    if runner is not None:
        return runner(args)
    if not gh_available():
        return 1, None
    try:
        proc = subprocess.run(
            ["gh", "api", "--method", "POST", *args],
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return 1, None
    except Exception:
        return 1, None
    body = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    if proc.returncode != 0:
        return proc.returncode, (body or None)
    if not body:
        return proc.returncode, {}
    try:
        return proc.returncode, json.loads(body)
    except Exception:
        return proc.returncode, body


def request_job_rerun(
    repo: str,
    job_id: int,
    *,
    runner: GHRunner = None,
) -> RetryResult:
    """POST /repos/{repo}/actions/jobs/{job_id}/rerun — minimal targeting."""
    args = [f"repos/{repo}/actions/jobs/{job_id}/rerun"]
    status, body = _gh_post(args, runner=runner)
    return RetryResult(
        accepted=bool(status) and 200 <= int(status) < 300,
        level=RETRY_JOB_LEVEL,
        status_code=int(status) if status else None,
        body=body,
        job_id=str(job_id),
        attempted_at=int(time.time()),
    )


def request_rerun_failed_jobs(
    repo: str,
    run_id: int,
    *,
    runner: GHRunner = None,
) -> RetryResult:
    """POST /repos/{repo}/actions/runs/{run_id}/rerun-failed-jobs — bounded
    workflow-level fallback used only when job-level rerun is unavailable."""
    args = [f"repos/{repo}/actions/runs/{run_id}/rerun-failed-jobs"]
    status, body = _gh_post(args, runner=runner)
    return RetryResult(
        accepted=bool(status) and 200 <= int(status) < 300,
        level=RETRY_WORKFLOW_LEVEL,
        status_code=int(status) if status else None,
        body=body,
        run_id=str(run_id),
        attempted_at=int(time.time()),
    )


def attempt_bounded_rerun(
    repo: str,
    *,
    queued_job_ids: list[str],
    run_id: Optional[str] = None,
    runner: GHRunner = None,
) -> RetryResult:
    """Attempt ONE bounded AUTO rerun per stall episode.

    Targeting: job-level rerun of the queued/stuck job(s) first. When no
    concrete job id is available (or the job-level endpoint is refused with a
    404/410-style "unavailable"), fall back to the affected workflow's
    ``rerun-failed-jobs`` — never a whole-PR rerun, never cancel/re-push.

    Any 4xx/5xx refusal returns ``accepted=False`` with the API evidence; the
    caller escalates (OWNER_ACTION_REQUIRED). A 403 on an ACTIVE run is the
    documented real-world outcome (GitHub refuses rerun of queued jobs while
    the workflow run is still running).
    """
    # Prefer concrete job-level rerun for each queued job (minimal target).
    for job_id in queued_job_ids:
        if not str(job_id).isdigit():
            continue
        result = request_job_rerun(repo, int(job_id), runner=runner)
        if result.accepted:
            return result
        # A 403/other refusal on job-level -> do NOT loop; try the bounded
        # workflow-level fallback once when a run id is known.
        if run_id and str(run_id).isdigit():
            fallback = request_rerun_failed_jobs(repo, int(run_id), runner=runner)
            if fallback.accepted:
                fallback.job_id = str(job_id)
                return fallback
            # Keep the richer refusal evidence (job-level first).
            return result
        return result
    if run_id and str(run_id).isdigit():
        return request_rerun_failed_jobs(repo, int(run_id), runner=runner)
    return RetryResult(
        accepted=False,
        level=RETRY_NONE,
        status_code=None,
        body={"error": "no concrete queued job id or run id available for retry"},
        attempted_at=int(time.time()),
    )


# ---------------------------------------------------------------------------
# Idempotent emission on a kanban card (comment + event)
# ---------------------------------------------------------------------------

# Event kind used for the CI watchdog (board gains only event/comment rows —
# Mission Control stays a read-only projection).
EXTERNAL_CI_WAIT_EVENT_KIND = "external_ci_wait"
EXTERNAL_CI_MARKER = "EXTERNAL-CI-WATCHDOG"


def _alert_fingerprint(alert: dict) -> str:
    """Stable fingerprint for idempotency: one alert per (state + evidence set).

    The fingerprint includes the internal CI state plus the concrete queued
    job ids so a transition (WAITING_LONG -> INFRA_STALLED) emits a fresh
    event while a repeated same-state tick stays silent.
    """
    raw = {
        "ciState": alert.get("ciState"),
        "queued": sorted(alert.get("queuedJobIds") or []),
        "failed": sorted(alert.get("failedJobIds") or []),
        "repo": alert.get("repo") or "",
        "pr": alert.get("prNumber"),
        "head": alert.get("headSha") or "",
        "captured": alert.get("capturedAt"),
    }
    payload = json.dumps(raw, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def last_external_ci_wait_event(
    conn: Any, task_id: str
) -> Optional[dict]:
    """Return the most recent ``external_ci_wait`` event payload on a card."""
    try:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id, EXTERNAL_CI_WAIT_EVENT_KIND),
        ).fetchone()
    except Exception:
        return None
    if not row or not row["payload"]:
        return None
    try:
        return json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
    except Exception:
        return None


def emit_external_ci_wait_alert(
    conn: Any,
    task_id: str,
    alert: dict,
    *,
    author: str = "ci-watchdog",
    body_override: Optional[str] = None,
) -> bool:
    """Persist one operator alert on a card (comment + event), idempotently.

    Idempotency: when the most recent ``external_ci_wait`` event on the card
    carries the same fingerprint (same CI state + same queued/failed job
    evidence), no duplicate comment/event is written. A changed state or a
    changed job set emits a fresh same-thread update (superseding the prior
    alert), matching the UX "1x per (workflow run + job set), replaced not
    stacked".

    The event payload carries the full 10-field alert plus the raw evidence
    fingerprint so Mission Control and QA can audit the classification.
    """
    try:
        from hermes_cli import kanban_db as kb
    except Exception:
        return False
    fingerprint = _alert_fingerprint(alert)
    prior = last_external_ci_wait_event(conn, task_id)
    if prior and (prior.get("fingerprint") or _alert_fingerprint(prior)) == fingerprint:
        return False

    body = body_override or render_alert_text(alert)
    marker_line = f"{EXTERNAL_CI_MARKER} — {alert.get('externalDependencyStatus')}"
    try:
        with kb.write_txn(conn):
            kb.add_comment(conn, task_id, author, f"{marker_line}\n{body}")
            kb._append_event(
                conn,
                task_id,
                EXTERNAL_CI_WAIT_EVENT_KIND,
                {
                    **alert,
                    "fingerprint": fingerprint,
                    "queuedJobIds": alert.get("queuedJobIds") or [],
                    "failedJobIds": alert.get("failedJobIds") or [],
                    "marker": EXTERNAL_CI_MARKER,
                },
            )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Retry episode state (ONE bounded retry per stall episode)
# ---------------------------------------------------------------------------

# Event kind recorded when a bounded AUTO retry is attempted (accepted or
# refused — both are audit evidence).
EXTERNAL_CI_RETRY_EVENT_KIND = "external_ci_retry"

# external_ci_wait ciState values that CLOSE a stall episode (execution
# resumed or the wait resolved): after one of these, a fresh stall episode may
# use its one-retry budget again.
_EPISODE_CLOSING_STATES = {RUNNING, COMPLETED, CI_FAILED}


def last_retry_attempt(conn: Any, task_id: str) -> Optional[dict]:
    """Most recent ``external_ci_retry`` payload (or None)."""
    try:
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id, EXTERNAL_CI_RETRY_EVENT_KIND),
        ).fetchone()
    except Exception:
        return None
    if not row or not row["payload"]:
        return None
    try:
        payload = (
            json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        )
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _episode_closed_since(conn: Any, task_id: str, since_ts: int) -> bool:
    """True when a resolving CI state was emitted after ``since_ts``.

    An episode (CI_INFRA_STALLED -> resolution) is closed by execution
    resuming (RUNNING), checks passing (COMPLETED) or a real failure
    (CI_FAILED). A closed episode releases the one-retry budget so a NEW
    stall episode can retry once more.
    """
    try:
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "AND created_at > ? ORDER BY id ASC",
            (task_id, EXTERNAL_CI_WAIT_EVENT_KIND, int(since_ts)),
        ).fetchall()
    except Exception:
        return False
    for row in rows:
        if not row["payload"]:
            continue
        try:
            payload = (
                json.loads(row["payload"])
                if isinstance(row["payload"], str)
                else row["payload"]
            )
        except Exception:
            continue
        if str((payload or {}).get("ciState") or "") in _EPISODE_CLOSING_STATES:
            return True
    return False


def retry_attempted_for_episode(
    conn: Any,
    task_id: str,
    *,
    now: Optional[int] = None,
    retry_window_seconds: int = DEFAULT_RETRY_WINDOW_MINUTES * 60,
) -> bool:
    """Return whether the ONE bounded retry of the current episode is spent.

    The budget is one attempt per stall episode: from CI_INFRA_STALLED entry
    until a resolving state (RUNNING/COMPLETED/CI_FAILED) closes the episode.
    A bare previous attempt (no resolving event after it) means the budget is
    spent — no retry loop can spin on subsequent ticks.
    """
    last = last_retry_attempt(conn, task_id)
    if last is None:
        return False
    attempted_at = int(last.get("attemptedAt") or 0)
    if _episode_closed_since(conn, task_id, attempted_at):
        return False
    return True


def retry_observation_expired(
    conn: Any,
    task_id: str,
    *,
    now: Optional[int] = None,
    retry_window_seconds: int = DEFAULT_RETRY_WINDOW_MINUTES * 60,
) -> bool:
    """Return whether an ACCEPTED bounded retry has stayed queued beyond the
    bounded observation window (retry_window_minutes, default 30).

    True only when: the most recent retry attempt was ACCEPTED (2xx), no
    resolving state (RUNNING/COMPLETED/CI_FAILED) arrived after it, and the
    window has elapsed. This is the ``retry remains queued -> Hermes exhausted
    its bounded attempt -> OWNER_ACTION_REQUIRED`` transition (criterion H).
    A refused attempt (never accepted) is not "in observation" — the refusal
    already escalated at attempt time.
    """
    last = last_retry_attempt(conn, task_id)
    if last is None or not last.get("accepted"):
        return False
    attempted_at = int(last.get("attemptedAt") or 0)
    if not attempted_at:
        return False
    if _episode_closed_since(conn, task_id, attempted_at):
        return False
    now_i = int(now if now is not None else time.time())
    return (now_i - attempted_at) > int(retry_window_seconds)


def record_retry_attempt(
    conn: Any,
    task_id: str,
    result: RetryResult,
) -> bool:
    """Persist a bounded retry attempt (accepted or refused) for audit."""
    try:
        from hermes_cli import kanban_db as kb
    except Exception:
        return False
    try:
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                EXTERNAL_CI_RETRY_EVENT_KIND,
                result.to_dict(),
            )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# High-level watchdog evaluation for one mission
# ---------------------------------------------------------------------------


def evaluate_external_ci_wait(
    conn: Any,
    task_id: str,
    snapshot: EvidenceSnapshot,
    *,
    config: Optional[dict] = None,
    policy: Optional[dict] = None,
    now: Optional[int] = None,
    auto_remediation: bool = True,
    runner: GHRunner = None,
) -> dict:
    """Run one watchdog evaluation for a mission card and persist alerts.

    Steps:
      1. Resolve thresholds (config kanban.ci_wait + per-mission override keyed
         by the umbrella task id, resolved via the kanban graph when the card
         has a mission parent).
      2. Classify from the evidence snapshot (fail-closed).
      3. Emit the operator alert comment/event idempotently (one per state +
         evidence set).
      4. When CI_INFRA_STALLED and bounded AUTO remediation is enabled and no
         retry was attempted for this episode: attempt ONE job-level rerun
         (workflow-level fallback only when job-level is unavailable). A 2xx
         acceptance records CI_RECOVERING; any API refusal records the refusal
         evidence and escalates the discussion payload to OWNER_ACTION_REQUIRED.

    Returns the classification + alert + retry result dict (no exception
    escapes for board-watcher callers — a broken watchdog must not break the
    dispatcher tick).
    """
    now_i = int(now if now is not None else time.time())
    out: dict = {"taskId": task_id, "evaluatedAt": now_i}

    # Resolve the mission umbrella id for per-mission overrides.
    umbrella_id: Optional[str] = None
    try:
        from hermes_cli import kanban_db as kb

        task = kb.get_task(conn, task_id)
        if task is not None:
            if task.role == "umbrella":
                umbrella_id = task.id
            else:
                parent = kb.find_mission_parent(conn, task_id)
                umbrella_id = parent.id if parent is not None else None
    except Exception:
        umbrella_id = None

    if policy is None:
        policy = ci_wait_policy_from_config(config)

    classification = classify_external_ci_wait(
        snapshot, policy=policy, now=now_i,
    )
    classification["repo"] = snapshot.repo
    classification["pr_number"] = snapshot.pr_number
    classification["head_sha"] = snapshot.head_sha
    classification["captured_at"] = snapshot.captured_at
    classification["queuedJobIds"] = [
        str((j or {}).get("id")) for j in snapshot.jobs if job_is_queued(j)
    ]
    classification["failedJobIds"] = [
        str((j or {}).get("id")) for j in snapshot.jobs if job_is_failed(j)
    ]
    out["classification"] = classification

    state = classification.get("ci_state")
    retry_available = False
    retry_result: Optional[dict] = None

    # Resolve the effective retry window (config / mission override) so the
    # post-retry observation window matches the operator policy.
    effective = resolve_ci_wait_policy(policy, umbrella_id)
    retry_window_seconds = int(effective["retry_window_minutes"]) * 60

    # Determine the queued run id (informational evidence for the retry call).
    queued_run_id: Optional[str] = None
    run_by_job: dict = {}
    for job in snapshot.jobs:
        rid = (job or {}).get("run_id")
        if rid is not None:
            run_by_job[str((job or {}).get("id"))] = str(rid)
    for jid in classification.get("queuedJobIds") or []:
        queued_run_id = run_by_job.get(jid) or queued_run_id

    # --- Post-retry observation overlay ------------------------------------
    # An ACCEPTED bounded retry opens an observation window: while the retried
    # job stays queued within retry_window_minutes the mission is RECOVERING
    # (KEEP_OPEN, OWNER ACTION NONE). When the window elapses with no
    # execution and no resolution, Hermes has exhausted its bounded attempt ->
    # OWNER_ACTION_REQUIRED. A refused attempt never enters observation — it
    # already escalated at attempt time. Execution/resolution states
    # (RUNNING/COMPLETED/CI_FAILED) from the live snapshot win over the
    # overlay: the episode closes and the classifier reports reality.
    last_retry = last_retry_attempt(conn, task_id)
    retry_in_observation = bool(
        last_retry
        and last_retry.get("accepted")
        and not _episode_closed_since(conn, task_id, int(last_retry.get("attemptedAt") or 0))
    )
    if (
        retry_in_observation
        and state in (CI_WAITING, CI_WAITING_LONG, CI_INFRA_STALLED)
    ):
        if retry_observation_expired(
            conn, task_id, now=now_i, retry_window_seconds=retry_window_seconds,
        ):
            classification["ci_state"] = OWNER_ACTION_REQUIRED
            classification["reason"] = (
                "Bounded AUTO rerun stayed queued beyond the observation "
                "window — Hermes exhausted its bounded attempt; "
                "OWNER_ACTION_REQUIRED."
            )
        else:
            classification["ci_state"] = CI_RECOVERING
            classification["reason"] = (
                "Bounded AUTO rerun accepted — still queued within the "
                "observation window; recovery in progress."
            )
        state = classification.get("ci_state")

    # --- ONE bounded AUTO retry per stall episode ---------------------------
    # UX sequence: emit the prominent CI_INFRA_STALLED alert FIRST (KEEP_OPEN,
    # OWNER ACTION NONE while a bounded retry is possible), then attempt the
    # retry and update the same thread to CI_RECOVERING / OWNER_ACTION_REQUIRED.
    stall_alert_emitted = False
    if (
        state == CI_INFRA_STALLED
        and classification.get("evidence_ok")
        and auto_remediation
        and not retry_attempted_for_episode(conn, task_id)
    ):
        # 2.2 prominent alert (retry about to be launched -> OWNER ACTION NONE).
        stall_alert = build_operator_alert(
            classification,
            mission_title="",
            pr_url="",
            retry_available=True,
            retry_evidence=None,
        )
        stall_alert["umbrellaTaskId"] = umbrella_id
        stall_alert["thresholds"] = {
            "warningMinutes": effective["warning_minutes"],
            "stallMinutes": effective["stall_minutes"],
            "retryWindowMinutes": effective["retry_window_minutes"],
            "warningEnabled": effective["warning_enabled"],
        }
        stall_alert_emitted = emit_external_ci_wait_alert(conn, task_id, stall_alert)

        result = attempt_bounded_rerun(
            snapshot.repo,
            queued_job_ids=classification.get("queuedJobIds") or [],
            run_id=queued_run_id,
            runner=runner,
        )
        record_retry_attempt(conn, task_id, result)
        retry_result = result.to_dict()
        if result.accepted:
            classification["ci_state"] = CI_RECOVERING
            classification["reason"] = (
                "Bounded AUTO job-level rerun accepted by the GitHub API — "
                "observing within the retry window."
            )
        else:
            classification["ci_state"] = OWNER_ACTION_REQUIRED
            classification["reason"] = (
                "Bounded AUTO rerun refused by the GitHub API (evidence: "
                f"status {result.status_code}) — Hermes cannot retry; "
                "OWNER_ACTION_REQUIRED."
            )
        state = classification.get("ci_state")
        retry_available = False
    elif state == CI_INFRA_STALLED:
        # No retry this tick: auto remediation is disabled OR the episode
        # budget is already spent (a previous attempt/refusal was recorded).
        # Hermes cannot recover bounded -> OWNER_ACTION_REQUIRED persists.
        classification["ci_state"] = OWNER_ACTION_REQUIRED
        classification["reason"] = (
            "Bounded AUTO rerun unavailable or already attempted this "
            "episode — Hermes cannot retry; OWNER_ACTION_REQUIRED."
        )
        state = OWNER_ACTION_REQUIRED
        retry_available = False

    alert = build_operator_alert(
        classification,
        mission_title="",
        pr_url="",
        retry_available=retry_available,
        retry_evidence=retry_result or (last_retry if last_retry else None),
    )
    alert["umbrellaTaskId"] = umbrella_id
    alert["thresholds"] = {
        "warningMinutes": effective["warning_minutes"],
        "stallMinutes": effective["stall_minutes"],
        "retryWindowMinutes": effective["retry_window_minutes"],
        "warningEnabled": effective["warning_enabled"],
    }

    # --- Emission policy (UX-1: CI_WAITING normal is SILENT) ----------------
    # A healthy in-window wait produces ZERO visible artefacts: no comment, no
    # notified event, no status flip. Only operator-visible transitions emit:
    #   * evidence-missing (fail-closed INFO signal, non-alarmist) — once per
    #     fingerprint (idempotent);
    #   * WAITING_LONG / INFRA_STALLED / RECOVERING / OWNER_ACTION_REQUIRED /
    #     CI_FAILED alerts;
    #   * RUNNING / COMPLETED resolution updates, but ONLY when an alert was
    #     already emitted on this card (close the same thread; never spam a
    #     mission that never alerted).
    emit_alert = True
    if state == CI_WAITING and classification.get("evidence_ok"):
        emit_alert = False  # normal healthy wait — silent (UX-1)
    elif state == CI_WAITING and not classification.get("evidence_ok"):
        # Evidence-missing: non-alarmist INFO signal (fail-closed) — emitted
        # idempotently so an API outage does not spam the thread.
        emit_alert = True
    elif state in (RUNNING, COMPLETED):
        emit_alert = last_external_ci_wait_event(conn, task_id) is not None
    if emit_alert:
        emitted = emit_external_ci_wait_alert(conn, task_id, alert)
    else:
        emitted = False
    out["alert"] = alert
    out["emitted"] = emitted
    out["stallAlertEmitted"] = stall_alert_emitted
    out["retry"] = retry_result or (last_retry if last_retry else None)
    out["umbrellaTaskId"] = umbrella_id
    return out


def run_external_ci_watchdog_for_pr(
    conn: Any,
    task_id: str,
    repo: str,
    pr_number: int,
    *,
    head_sha: Optional[str] = None,
    config: Optional[dict] = None,
    policy: Optional[dict] = None,
    runner: GHRunner = None,
    auto_remediation: bool = True,
) -> dict:
    """One-shot watchdog evaluation for a mission that waits on a PR's checks.

    This is the primary ACTIVATION hook (Product OQ-4): the owning session
    calls it when it knows it is waiting on required CI checks for a PR. It
    collects a fresh GitHub evidence snapshot (read-only REST) and runs
    :func:`evaluate_external_ci_wait`, which classifies and persists alerts.

    Returns the same dict as :func:`evaluate_external_ci_wait` plus the
    collected snapshot; ``snapshot.evidence_complete()`` tells the caller
    whether live evidence was obtainable (fail-closed when not).
    """
    snapshot = collect_external_ci_snapshot(
        repo, pr_number, head_sha=head_sha, runner=runner,
    )
    out = evaluate_external_ci_wait(
        conn,
        task_id,
        snapshot,
        config=config,
        policy=policy,
        runner=runner,
        auto_remediation=auto_remediation,
    )
    out["snapshot"] = snapshot.to_dict()
    return out
