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
import re
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


def _required_pair_map(required_checks: Any) -> Optional[dict]:
    """Map each required job id to its single authoritative run id.

    Built strictly from the GraphQL required-check rollup rows (each row
    carries ``jobId`` + ``runId``). Returns None when the rollup is absent, a
    row is malformed, or the same job id appears more than once (duplicate or
    conflicting rows) — callers MUST fail closed on None, because a retry
    target is only ever derived from an exact, unique (jobId, runId) pair.
    """
    if not isinstance(required_checks, list) or not required_checks:
        return None
    pairs: dict[str, str] = {}
    for check in required_checks:
        if not isinstance(check, dict):
            return None
        job_id = str(check.get("jobId") or "")
        run_id = str(check.get("runId") or "")
        if not job_id.isdigit() or not run_id.isdigit():
            return None
        if job_id in pairs:
            # Duplicate/conflicting required row — same job bound twice.
            return None
        pairs[job_id] = run_id
    return pairs


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
    # Authoritative required-check proof from GraphQL CheckRun.isRequired for
    # this PR number and head commit.  ``required=False`` is meaningful only
    # when this proof was collected successfully.
    required_check_evidence: bool = False
    required_head_sha: str = ""
    required_job_ids: list[str] = field(default_factory=list)
    selected_run_ids: list[str] = field(default_factory=list)
    required_checks: list[dict] = field(default_factory=list)
    # A newer workflow run (same workflow+branch/PR, or a newer SHA) exists.
    superseded: bool = False
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "EvidenceSnapshot":
        data = data or {}
        try:
            captured_at = int(data.get("captured_at") or 0)
        except (TypeError, ValueError):
            captured_at = 0
        required_job_ids = data.get("required_job_ids")
        selected_run_ids = data.get("selected_run_ids")
        return cls(
            captured_at=captured_at,
            repo=str(data.get("repo") or ""),
            pr_number=data.get("pr_number"),
            head_sha=str(data.get("head_sha") or ""),
            runs=list(data.get("runs") or []) if isinstance(data.get("runs"), list) else [],
            jobs=list(data.get("jobs") or []) if isinstance(data.get("jobs"), list) else [],
            required=bool(data.get("required")),
            required_check_evidence=bool(data.get("required_check_evidence")),
            required_head_sha=str(data.get("required_head_sha") or ""),
            required_job_ids=(
                [str(v) for v in required_job_ids]
                if isinstance(required_job_ids, list)
                else []
            ),
            selected_run_ids=(
                [str(v) for v in selected_run_ids]
                if isinstance(selected_run_ids, list)
                else []
            ),
            required_checks=(
                list(data.get("required_checks") or [])
                if isinstance(data.get("required_checks"), list)
                else []
            ),
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
            "required_check_evidence": self.required_check_evidence,
            "required_head_sha": self.required_head_sha,
            "required_job_ids": self.required_job_ids,
            "selected_run_ids": self.selected_run_ids,
            "required_checks": self.required_checks,
            "superseded": self.superseded,
        }

    def authoritative_pairs(self) -> Optional[dict]:
        """Authoritative (jobId -> runId) map from the GraphQL rollup.

        None means the rollup is missing/duplicated/conflicting — no retry
        target may be derived from it (fail closed).
        """
        return _required_pair_map(self.required_checks)

    def evidence_complete(self) -> bool:
        """Fail-closed gate: is the snapshot complete enough to classify?

        A stall classification requires every evidence field: a fresh
        timestamp, a named repo/PR/SHA, at least one run, and jobs for that
        run with steps visible. Missing any of these -> incomplete.

        Correlation is exact-pair, never set-based: the authoritative GraphQL
        rollup binds each required job id to ONE run id. Crossed pairs (job
        row run_id disagrees with the rollup pair), duplicate/conflicting
        rows, missing/extra pairs and endpoint/job run mismatches all fail
        closed here so no later write can be derived from ambiguous evidence.
        """
        if (
            not isinstance(self.captured_at, int)
            or isinstance(self.captured_at, bool)
            or self.captured_at <= 0
            or not self.repo
            or self.repo.count("/") != 1
        ):
            return False
        try:
            pr_number = int(self.pr_number or 0)
        except (TypeError, ValueError):
            return False
        if pr_number <= 0 or not self.head_sha:
            return False
        if not self.required_check_evidence:
            return False
        if not self.required_head_sha or self.required_head_sha != self.head_sha:
            return False
        # An authoritative rollup with no required Action check proves that an
        # optional queued workflow must not be classified or retried.
        if not self.required:
            return not self.required_job_ids
        required_ids = {str(v) for v in self.required_job_ids}
        selected_run_ids = {str(v) for v in self.selected_run_ids}
        if not required_ids or not selected_run_ids or not self.runs or not self.jobs:
            return False
        if not all(value.isdigit() for value in required_ids | selected_run_ids):
            return False

        valid_statuses = {"queued", "in_progress", "completed", "waiting", "pending", "requested"}
        valid_conclusions = {
            None, "success", "failure", "neutral", "cancelled", "skipped",
            "timed_out", "action_required", "stale", "startup_failure",
        }
        if not self.required_checks:
            return False

        # Authoritative rollup: exactly one unique (jobId -> runId) pair per
        # required job. Duplicate/conflicting rows, or pairs referencing ids
        # outside the declared required/selected sets, fail closed.
        pairs = self.authoritative_pairs()
        if pairs is None:
            return False
        if set(pairs) != required_ids or set(pairs.values()) != selected_run_ids:
            return False
        for check in self.required_checks:
            if not isinstance(check, dict) or check.get("isRequired") is not True:
                return False
            job_id = str(check.get("jobId") or "")
            run_id = str(check.get("runId") or "")
            if pairs.get(job_id) != run_id:
                return False
            if job_id not in required_ids or run_id not in selected_run_ids:
                return False
            if check.get("status") not in valid_statuses or "conclusion" not in check:
                return False
            if check.get("conclusion") not in valid_conclusions:
                return False

        # REST run rows: every selected run id exactly once (duplicate run
        # rows fail closed), bound to the current head sha.
        run_rows: dict[str, int] = {}
        for run in self.runs:
            if not isinstance(run, dict):
                return False
            run_id = str(run.get("id") or "")
            if not run_id.isdigit() or run_id not in selected_run_ids:
                return False
            if run.get("head_sha") != self.head_sha:
                return False
            if run.get("status") not in valid_statuses or "conclusion" not in run:
                return False
            if run.get("conclusion") not in valid_conclusions:
                return False
            if _parse_github_ts(run.get("created_at")) is None:
                return False
            run_rows[run_id] = run_rows.get(run_id, 0) + 1
        if set(run_rows) != selected_run_ids or any(
            count != 1 for count in run_rows.values()
        ):
            return False

        # REST job rows: every required job exactly once, and each row must be
        # bound to its OWN authoritative run id. A crossed pair (row run_id
        # disagrees with the rollup pair for that job) or a duplicate row
        # fails closed — no later write derives from ambiguous evidence.
        job_rows: dict[str, int] = {}
        for job in self.jobs:
            if not isinstance(job, dict):
                return False
            job_id = str(job.get("id") or "")
            run_id = str(job.get("run_id") or "")
            if job_id not in required_ids or not job_id.isdigit():
                return False
            if run_id != pairs.get(job_id) or not run_id.isdigit():
                return False
            if job.get("status") not in valid_statuses or "conclusion" not in job:
                return False
            if job.get("conclusion") not in valid_conclusions:
                return False
            if "steps" not in job or not isinstance(job.get("steps"), list):
                return False
            if _parse_github_ts(job.get("started_at")) is None:
                return False
            if job.get("status") == "completed" and _parse_github_ts(job.get("completed_at")) is None:
                return False
            job_rows[job_id] = job_rows.get(job_id, 0) + 1
        return set(job_rows) == required_ids and all(
            count == 1 for count in job_rows.values()
        )


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
    if _job_status(job) != "completed" or _job_conclusion(job) != "failure":
        return False
    # A conclusion by itself is not proof of execution. Require the step
    # records exposed for an executed Action job.
    return job_execution_started(job)


def job_is_queued(job: dict) -> bool:
    return _job_status(job) == "queued" and _job_conclusion(job) is None


def _run_created_at(run: dict) -> Optional[int]:
    return _parse_github_ts((run or {}).get("created_at"))


def _required_jobs(snapshot: EvidenceSnapshot) -> list[dict]:
    required_ids = {str(value) for value in snapshot.required_job_ids}
    return [
        job for job in snapshot.jobs
        if isinstance(job, dict) and str(job.get("id")) in required_ids
    ]


def _max_queue_minutes(snapshot: EvidenceSnapshot, now: int) -> int:
    """Longest observed queue wait among required queued jobs.

    Queue duration measures from the job's queued_at / run created_at up to
    the snapshot captured_at (never from wall-clock alone).
    """
    captured = snapshot.captured_at or now
    run_by_id = {str((r or {}).get("id")): r for r in snapshot.runs}
    longest = 0
    for job in _required_jobs(snapshot):
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
        for j in _required_jobs(snapshot)
        if isinstance(j, dict) and job_is_queued(j)
    ]


def _snapshot_failed_job_ids(snapshot: EvidenceSnapshot) -> list[str]:
    return [
        str((j or {}).get("id"))
        for j in _required_jobs(snapshot)
        if isinstance(j, dict) and job_is_failed(j)
    ]


def _current_run_id(snapshot: EvidenceSnapshot) -> Optional[str]:
    """Id of the newest observed run (the one being evaluated)."""
    selected = {str(value) for value in snapshot.selected_run_ids}
    runs = [
        run for run in snapshot.runs
        if isinstance(run, dict) and str(run.get("id")) in selected
    ]
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

    # CI_FAILED requires real job-level execution proof then failure. A run
    # conclusion alone can be inconsistent or belong to an optional sibling;
    # it never fabricates a failure for the selected required job set.
    failed_jobs = _snapshot_failed_job_ids(snapshot)
    if failed_jobs:
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
        job_is_running(job) for job in _required_jobs(snapshot)
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
    # Produit §1.5 / §6-F (prime rule Produit > UX, UX reserve 3): recovery is
    # a bounded, journaled INFO signal — never the WARNING of a prolonged wait.
    CI_RECOVERING: "INFO",
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
    if state == CI_WAITING and not evidence_ok:
        # Evidence-missing signal (fail-closed): non-alarmist INFO per Produit
        # §2 — it must never inherit the silent NONE of a healthy wait (a
        # healthy CI_WAITING with evidence_ok=True stays NONE, zero events).
        attention = "INFO"
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
        "repo": repo,
        "prNumber": pr,
        "headSha": classification.get("head_sha") or "",
        "runIds": sorted(str(value) for value in (classification.get("runIds") or [])),
        "queuedJobIds": sorted(
            str(value) for value in (classification.get("queuedJobIds") or [])
        ),
        "failedJobIds": sorted(
            str(value) for value in (classification.get("failedJobIds") or [])
        ),
        "correlation": classification.get("correlation") or {},
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
#: or a parsed JSON value (dict/list/None). Malformed injected statuses are
#: accepted by the type so the production path can fail closed on them.
GHRunner = Optional[Callable[[list[str]], "tuple[Any, Any]"]]

_REQUIRED_CHECKS_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      headRefOid
      commits(last: 1) {
        nodes {
          commit {
            statusCheckRollup {
              contexts(first: 100) {
                pageInfo { hasNextPage }
                nodes {
                  __typename
                  ... on CheckRun {
                    databaseId
                    name
                    status
                    conclusion
                    isRequired(pullRequestNumber: $number)
                    checkSuite { workflowRun { databaseId } }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
""".strip()


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
    pr_head_sha = str(pr.get("head", {}).get("sha") or "")
    raw["head_sha"] = pr_head_sha
    raw["pr_open"] = str(pr.get("state") or "") == "open"
    if not pr_head_sha or (head_sha and str(head_sha) != pr_head_sha):
        return EvidenceSnapshot.from_dict(raw)

    # Authoritative authorization boundary: CheckRun.isRequired is evaluated
    # by GitHub for this exact pull request. PR-open alone never grants retry
    # permission. The rollup is anchored to headRefOid and exposes the Action
    # job/run database IDs used for REST correlation below.
    try:
        owner, name = repo.split("/", 1)
    except ValueError:
        return EvidenceSnapshot.from_dict(raw)
    status, required_rollup = _gh_json(
        [
            "graphql",
            "-f", f"owner={owner}",
            "-f", f"name={name}",
            "-F", f"number={int(pr_number)}",
            "-f", f"query={_REQUIRED_CHECKS_QUERY}",
        ],
        runner=runner,
    )
    try:
        pull_request = required_rollup["data"]["repository"]["pullRequest"]
        rollup_head = str(pull_request["headRefOid"] or "")
        contexts = pull_request["commits"]["nodes"][0]["commit"][
            "statusCheckRollup"
        ]["contexts"]
        has_next_page = contexts["pageInfo"]["hasNextPage"]
        contexts = contexts["nodes"]
    except (KeyError, IndexError, TypeError):
        return EvidenceSnapshot.from_dict(raw)
    if (
        status != 0
        or rollup_head != pr_head_sha
        or has_next_page is not False
        or not isinstance(contexts, list)
    ):
        return EvidenceSnapshot.from_dict(raw)

    required_checks: list[dict] = []
    malformed_required = False
    for context in contexts:
        if not isinstance(context, dict) or context.get("__typename") != "CheckRun":
            continue
        if context.get("isRequired") is not True:
            continue
        workflow_run = (context.get("checkSuite") or {}).get("workflowRun") or {}
        job_id = str(context.get("databaseId") or "")
        run_id = str(workflow_run.get("databaseId") or "")
        if not job_id.isdigit() or not run_id.isdigit():
            malformed_required = True
            continue
        required_checks.append(
            {
                "jobId": job_id,
                "runId": run_id,
                "name": str(context.get("name") or ""),
                "status": str(context.get("status") or "").lower(),
                "conclusion": (
                    str(context.get("conclusion")).lower()
                    if context.get("conclusion") is not None
                    else None
                ),
                "isRequired": True,
            }
        )
    if malformed_required:
        return EvidenceSnapshot.from_dict(raw)
    raw["required_check_evidence"] = True
    raw["required_head_sha"] = rollup_head
    raw["required_checks"] = required_checks
    # Authoritative exact pairs, never independent sets: when the rollup lists
    # a required check it must bind each job to exactly one run. A duplicate
    # or conflicting required row (same job bound twice) fails closed here so
    # no later REST row can be normalized into a crossed pair.
    if required_checks and _required_pair_map(required_checks) is None:
        return EvidenceSnapshot.from_dict(raw)
    authoritative_pairs = _required_pair_map(required_checks) or {}
    raw["required_job_ids"] = sorted(authoritative_pairs)
    raw["selected_run_ids"] = sorted(
        {run_id for run_id in authoritative_pairs.values()}
    )
    raw["required"] = bool(raw["pr_open"] and authoritative_pairs)
    # No required Action check exists for this head. This is complete evidence
    # for a silent, non-retrying exit even if an optional workflow is queued.
    if not raw["required"]:
        return EvidenceSnapshot.from_dict(raw)

    status, runs = _gh_json(
        [
            f"repos/{repo}/actions/runs?head_sha={pr_head_sha}&per_page=100",
            "--jq", ".workflow_runs",
        ],
        runner=runner,
    )
    if status != 0 or not isinstance(runs, list):
        return EvidenceSnapshot.from_dict(raw)
    selected_ids = set(raw["selected_run_ids"])
    selected_runs = [
        run for run in runs
        if isinstance(run, dict)
        and str(run.get("id") or "") in selected_ids
        and str(run.get("head_sha") or "") == pr_head_sha
    ]
    if {str(run.get("id")) for run in selected_runs} != selected_ids:
        return EvidenceSnapshot.from_dict(raw)
    raw["runs"] = selected_runs
    raw["superseded"] = False

    # Fetch every Action run referenced by the authoritative required-check
    # rollup, then retain exactly the required job rows — each one must come
    # from ITS authoritative run and declare that same run id (endpoint/job
    # run binding). A required job found under a different run, or appearing
    # twice, is crossed/duplicate evidence and fails closed here. Missing
    # fields remain missing; normalization must never turn absent ``steps``
    # into proof of 0.
    normalized_jobs: list[dict] = []
    seen_job_ids: set[str] = set()
    for run_id in sorted(selected_ids):
        status, jobs = _gh_json(
            [
                f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
                "--jq", ".jobs",
            ],
            runner=runner,
        )
        if status != 0 or not isinstance(jobs, list):
            return EvidenceSnapshot.from_dict(raw)
        for job in jobs:
            if not isinstance(job, dict) or str(job.get("id") or "") not in authoritative_pairs:
                continue
            job_id = str(job.get("id") or "")
            rest_run_id = str(job.get("run_id") or "")
            if (
                job_id in seen_job_ids
                or rest_run_id != run_id
                or authoritative_pairs[job_id] != rest_run_id
            ):
                # Duplicate required job row or endpoint/job run mismatch —
                # the REST row does not match the authoritative pair.
                return EvidenceSnapshot.from_dict(raw)
            seen_job_ids.add(job_id)
            normalized_jobs.append(
                {
                    key: job.get(key)
                    for key in (
                        "id", "run_id", "name", "status", "conclusion",
                        "started_at", "completed_at", "steps", "html_url",
                    )
                    if key in job
                }
            )
    if seen_job_ids != set(authoritative_pairs):
        # A required job id was never observed in its authoritative run's
        # jobs payload — missing pair evidence, fail closed.
        return EvidenceSnapshot.from_dict(raw)
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
RETRY_PERSISTED_BODY_MAX_CHARS = 512


def _http_status(value: Any) -> Optional[int]:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return None
    return status if 100 <= status <= 599 else None


def _sanitize_retry_text(value: Any) -> str:
    text = str(value or "").replace("\x00", " ")
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+)\b",
        "[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password)\s*[=:]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
    text = " ".join(text.split())
    if len(text) > RETRY_PERSISTED_BODY_MAX_CHARS:
        text = text[: RETRY_PERSISTED_BODY_MAX_CHARS - 1] + "…"
    return text


def _sanitize_retry_body(body: Any) -> Any:
    if body is None:
        return None
    if isinstance(body, dict):
        sanitized = {}
        for key in ("message", "error", "status", "documentation_url"):
            if key in body and body.get(key) is not None:
                sanitized[key] = _sanitize_retry_text(body.get(key))
        return sanitized or None
    return _sanitize_retry_text(body)


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
            "body": _sanitize_retry_body(self.body),
            "jobId": self.job_id,
            "runId": self.run_id,
            "attemptedAt": self.attempted_at,
        }


def _gh_post(
    args: list[str],
    *,
    runner: GHRunner = None,
) -> tuple[Optional[int], Any]:
    """POST via gh api. Returns (status_code, parsed_body_or_text)."""
    if runner is not None:
        status, body = runner(args)
        return _http_status(status), _sanitize_retry_body(body)
    if not gh_available():
        return None, None
    try:
        proc = subprocess.run(
            ["gh", "api", "--include", "--method", "POST", *args],
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return None, None
    except Exception:
        return None, None

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    stdout_status_matches = re.findall(r"(?im)^HTTP/\S+\s+(\d{3})\b", stdout)
    status_matches = stdout_status_matches or re.findall(
        r"(?i)\bHTTP\s+(\d{3})\b", stderr
    )
    status = _http_status(status_matches[-1]) if status_matches else None
    # gh exit 0 proves an accepted API response. --include normally preserves
    # the exact HTTP code; normalize to 200 only for older gh builds that omit
    # headers rather than confusing the OS return code with HTTP status.
    if proc.returncode == 0 and status is None:
        status = 200
    if proc.returncode != 0 and status is None:
        return None, _sanitize_retry_body(stderr or stdout)

    body_text = stdout or stderr
    if stdout_status_matches:
        sections = re.split(r"\r?\n\r?\n", stdout)
        body_text = sections[-1] if len(sections) > 1 else ""
    body_text = body_text.strip()
    if not body_text:
        return status, None
    try:
        return status, _sanitize_retry_body(json.loads(body_text))
    except Exception:
        return status, _sanitize_retry_body(body_text)


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
        accepted=status is not None and 200 <= status < 300,
        level=RETRY_JOB_LEVEL,
        status_code=status,
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
        accepted=status is not None and 200 <= status < 300,
        level=RETRY_WORKFLOW_LEVEL,
        status_code=status,
        body=body,
        run_id=str(run_id),
        attempted_at=int(time.time()),
    )


def attempt_bounded_rerun(
    repo: str,
    *,
    queued_job_ids: list[str],
    job_run_ids: Optional[dict] = None,
    run_id: Optional[str] = None,
    runner: GHRunner = None,
) -> RetryResult:
    """Attempt ONE bounded AUTO rerun per stall episode.

    Targeting: job-level rerun of the first concrete queued job (minimal
    target — exactly one job POST per episode). When that job's endpoint is
    refused with an explicit 404/410-style "unavailable", fall back to the
    affected workflow's ``rerun-failed-jobs`` bound to THAT JOB'S OWN run id
    (the authoritative pair from ``job_run_ids``) — never to another selected
    run. ``run_id`` is the legacy single-run binding used only when no
    per-job map is supplied; when ``job_run_ids`` is provided it is
    authoritative, and a job without a mapped run never triggers a workflow
    fallback (missing-pair evidence fails closed). Never a whole-PR rerun,
    never cancel/re-push.

    Any 4xx/5xx refusal returns ``accepted=False`` with the API evidence after
    exactly one job-level POST; a 403 on an ACTIVE run is the documented
    real-world outcome (GitHub refuses rerun of queued jobs while the
    workflow run is still running).
    """
    # Prefer concrete job-level rerun (minimal target): the first concrete
    # queued job. Auth/refusal/validation/server/timeout or a malformed
    # response stops after exactly this one job-level write.
    for job_id in queued_job_ids:
        if not str(job_id).isdigit():
            continue
        result = request_job_rerun(repo, int(job_id), runner=runner)
        if result.accepted:
            return result
        # Only an explicit endpoint-unavailable response authorizes the wider
        # workflow-level fallback, and only on the JOB'S OWN run (exact pair).
        own_run_id: Optional[str] = None
        if job_run_ids is not None:
            own_run_id = job_run_ids.get(str(job_id))
        else:
            own_run_id = run_id
        if result.status_code in {404, 410} and own_run_id and str(own_run_id).isdigit():
            fallback = request_rerun_failed_jobs(repo, int(own_run_id), runner=runner)
            if fallback.accepted:
                fallback.job_id = str(job_id)
                return fallback
            # Keep the richer refusal evidence (job-level first).
            return result
        return result
    # No concrete queued job id available: workflow-level fallback only when
    # the caller supplied an explicit single run id (legacy single-run call).
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
        "runs": sorted(alert.get("runIds") or []),
        "repo": alert.get("repo") or "",
        "pr": alert.get("prNumber"),
        "head": alert.get("headSha") or "",
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
    body = body_override or render_alert_text(alert)
    marker_line = f"{EXTERNAL_CI_MARKER} — {alert.get('externalDependencyStatus')}"
    try:
        with kb.write_txn(conn):
            # The read and write share one BEGIN IMMEDIATE transaction so two
            # evaluators cannot both emit the same stable evidence fingerprint.
            prior = last_external_ci_wait_event(conn, task_id)
            if prior and (
                prior.get("fingerprint") or _alert_fingerprint(prior)
            ) == fingerprint:
                return False
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


def _latest_episode_closure_id(conn: Any, task_id: str) -> int:
    """Return the newest event id that closes a retry episode."""
    try:
        rows = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id DESC",
            (task_id, EXTERNAL_CI_WAIT_EVENT_KIND),
        ).fetchall()
    except Exception:
        return 0
    for row in rows:
        try:
            payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        except Exception:
            continue
        if str((payload or {}).get("ciState") or "") in _EPISODE_CLOSING_STATES:
            return int(row["id"])
    return 0


def _retry_episode_key(
    task_id: str,
    snapshot: EvidenceSnapshot,
    classification: dict,
    closure_event_id: int,
) -> str:
    raw = {
        "task": task_id,
        "repo": snapshot.repo,
        "pr": snapshot.pr_number,
        "head": snapshot.head_sha,
        "runs": sorted(snapshot.selected_run_ids),
        "jobs": sorted(classification.get("queuedJobIds") or []),
        "generation": int(closure_event_id),
    }
    encoded = json.dumps(raw, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def reserve_retry_attempt(
    conn: Any,
    task_id: str,
    snapshot: EvidenceSnapshot,
    classification: dict,
    *,
    attempted_at: int,
    run_id: Optional[str],
) -> Optional[dict]:
    """Atomically reserve the one write allowed for the current stall episode."""
    try:
        from hermes_cli import kanban_db as kb
    except Exception:
        return None
    try:
        with kb.write_txn(conn):
            closure_id = _latest_episode_closure_id(conn, task_id)
            existing = conn.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? "
                "AND id > ? LIMIT 1",
                (task_id, EXTERNAL_CI_RETRY_EVENT_KIND, closure_id),
            ).fetchone()
            if existing is not None:
                return None
            episode_key = _retry_episode_key(
                task_id, snapshot, classification, closure_id,
            )
            payload = {
                "reservationState": "reserved",
                "episodeKey": episode_key,
                "accepted": False,
                "level": RETRY_NONE,
                "statusCode": None,
                "body": None,
                "jobId": next(iter(classification.get("queuedJobIds") or []), None),
                "runId": run_id,
                "attemptedAt": int(attempted_at),
                "correlation": {
                    "repo": snapshot.repo,
                    "prNumber": snapshot.pr_number,
                    "headSha": snapshot.head_sha,
                    "runIds": sorted(snapshot.selected_run_ids),
                    "requiredJobIds": sorted(snapshot.required_job_ids),
                },
            }
            kb._append_event(
                conn, task_id, EXTERNAL_CI_RETRY_EVENT_KIND, payload,
            )
            row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
            return {
                "eventId": int(row["id"]),
                "episodeKey": episode_key,
                "payload": payload,
            }
    except Exception:
        return None


def finalize_retry_attempt(
    conn: Any,
    task_id: str,
    reservation: dict,
    result: RetryResult,
) -> bool:
    """Finalize exactly the durable reservation created before the POST."""
    try:
        from hermes_cli import kanban_db as kb
    except Exception:
        return False
    payload = {
        **(reservation.get("payload") or {}),
        **result.to_dict(),
        "reservationState": "finalized",
        "episodeKey": reservation.get("episodeKey"),
    }
    try:
        with kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ? AND task_id = ? "
                "AND kind = ?",
                (
                    json.dumps(payload, ensure_ascii=False),
                    int(reservation.get("eventId") or 0),
                    task_id,
                    EXTERNAL_CI_RETRY_EVENT_KIND,
                ),
            )
            return cur.rowcount == 1
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

    # Resolve the umbrella override BEFORE classification. Reported thresholds
    # and behavior must be the same effective policy.
    effective = resolve_ci_wait_policy(policy, umbrella_id)
    classification = classify_external_ci_wait(
        snapshot, policy=effective, now=now_i,
    )
    classification["repo"] = snapshot.repo
    classification["pr_number"] = snapshot.pr_number
    classification["head_sha"] = snapshot.head_sha
    classification["captured_at"] = snapshot.captured_at
    classification["runIds"] = sorted(snapshot.selected_run_ids)
    classification["queuedJobIds"] = _snapshot_queued_job_ids(snapshot)
    classification["failedJobIds"] = _snapshot_failed_job_ids(snapshot)
    classification["correlation"] = {
        "requiredCheckEvidence": snapshot.required_check_evidence,
        "requiredHeadSha": snapshot.required_head_sha,
        "requiredJobIds": sorted(snapshot.required_job_ids),
        "selectedRunIds": sorted(snapshot.selected_run_ids),
        "requiredChecks": snapshot.required_checks,
    }
    out["classification"] = classification

    state = classification.get("ci_state")
    retry_available = False
    retry_result: Optional[dict] = None

    retry_window_seconds = int(effective["retry_window_minutes"]) * 60

    # Bind retry writes to the authoritative (jobId -> runId) pairs. Each
    # queued job belongs to exactly one run; the workflow-level fallback for
    # a job must target THAT job's own run (exact pair), never another
    # selected run. ``evidence_complete`` already guaranteed the REST rows
    # agree with the rollup pairs, so this map is the single source of truth.
    queued_job_ids = list(classification.get("queuedJobIds") or [])
    job_run_ids: dict = snapshot.authoritative_pairs() or {}
    first_queued_job_id = next(iter(queued_job_ids), None)
    queued_run_id = (
        job_run_ids.get(str(first_queued_job_id))
        if first_queued_job_id is not None
        else None
    )

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
    ):
        reservation = reserve_retry_attempt(
            conn,
            task_id,
            snapshot,
            classification,
            attempted_at=now_i,
            run_id=queued_run_id,
        )
        if reservation is not None:
            # Prominent alert is visible before the network write, while the
            # hidden atomic reservation already prevents another evaluator.
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
                queued_job_ids=queued_job_ids,
                job_run_ids=job_run_ids,
                run_id=queued_run_id,
                runner=runner,
            )
            finalize_retry_attempt(conn, task_id, reservation, result)
            retry_result = {
                **result.to_dict(),
                "reservationState": "finalized",
                "episodeKey": reservation.get("episodeKey"),
            }
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
        else:
            last_retry = last_retry_attempt(conn, task_id)
            if last_retry and last_retry.get("reservationState") == "reserved":
                classification["reason"] = (
                    "Bounded AUTO rerun reservation is in flight; no concurrent "
                    "GitHub write is authorized."
                )
                retry_available = True
            else:
                classification["ci_state"] = OWNER_ACTION_REQUIRED
                classification["reason"] = (
                    "Bounded AUTO rerun already attempted this episode — Hermes "
                    "cannot retry; OWNER_ACTION_REQUIRED."
                )
                state = OWNER_ACTION_REQUIRED
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
