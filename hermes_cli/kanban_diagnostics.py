"""Kanban diagnostics — structured, actionable distress signals for tasks.

A ``Diagnostic`` is a machine-readable description of something that's wrong
with a kanban task: a hallucinated card id, a spawn crash-loop, a task
stuck blocked for too long, etc. Each one carries:

* A **kind** (canonical code; UI/tests match on this).
* A **severity** (``warning`` / ``error`` / ``critical``).
* A **title** (one-line human description) and **detail** (longer text).
* A list of **suggested actions** — structured entries the dashboard
  turns into buttons and the CLI turns into hints.

Rules run over (task, recent events, recent runs, optional graph context) and
emit diagnostics. They are stateless and read-only — no DB writes. Callers compute
diagnostics on demand (on ``/board`` load, ``/tasks/:id`` fetch, or
``hermes kanban diagnostics``).

Design goals:

* Fixable-on-the-operator's-side signals only (missing config, phantom
  ids, crash loop). Not "the provider returned 502 once" — that's a
  transient runtime blip, not a diagnostic.
* Recoverable: every diagnostic comes with at least one suggested
  recovery action the operator can actually take from the UI.
* Auto-clearing: when the underlying failure mode resolves (a clean
  ``completed`` event arrives, a spawn succeeds, the task gets
  unblocked), the diagnostic stops firing. The audit event trail stays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional
import json
import time


# Severity rungs, ordered least → most urgent. The UI colors them
# amber (warning), orange (error), red (critical). Sorted outputs put
# critical first so operators see the worst fires at the top.
SEVERITY_ORDER = ("warning", "error", "critical")


def severity_at_or_above(severity: Optional[str], threshold: Optional[str]) -> bool:
    """Return True when ``severity`` meets or exceeds ``threshold``."""
    if threshold is None:
        return True
    if severity not in SEVERITY_ORDER or threshold not in SEVERITY_ORDER:
        return False
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(threshold)


# ---------------------------------------------------------------------------
# Operator attention axis (Operator Diagnostics Clarity contract)
#
# Every Diagnostic carries TWO orthogonal axes:
#   * technical severity (warning/error/critical, above) — unchanged,
#     used for triage order and legacy colouring;
#   * operator attention (NONE/INFO/WARNING/ACTION_REQUIRED/CRITICAL) —
#     what the OPERATOR should do about it. A diagnostic may exist with
#     NONE/INFO attention (e.g. a healthy, legitimately-queued task); its
#     emission never implies the operator must act.
# ---------------------------------------------------------------------------
ATTENTION_NONE = "NONE"
ATTENTION_INFO = "INFO"
ATTENTION_WARNING = "WARNING"
ATTENTION_ACTION_REQUIRED = "ACTION_REQUIRED"
ATTENTION_CRITICAL = "CRITICAL"
ATTENTION_ORDER = (
    ATTENTION_NONE,
    ATTENTION_INFO,
    ATTENTION_WARNING,
    ATTENTION_ACTION_REQUIRED,
    ATTENTION_CRITICAL,
)

# Owner action: REQUIRED exactly when attention is ACTION_REQUIRED/CRITICAL.
# WARNING = recommended action, never a hard requirement.
OWNER_ACTION_NONE = "NONE"
OWNER_ACTION_REQUIRED = "REQUIRED"

# Auto-recovery lifecycle states reflected by diagnostics. The ENGINE only
# renders states evidenced by events/board context — it never executes the
# recovery itself (that belongs to the dispatcher / stall watchdog lane).
AUTO_RECOVERY_NONE = "none"
AUTO_RECOVERY_IN_PROGRESS = "in_progress"
AUTO_RECOVERY_SUCCEEDED = "succeeded"
AUTO_RECOVERY_FAILED = "failed"
AUTO_RECOVERY_STATES = (
    AUTO_RECOVERY_NONE,
    AUTO_RECOVERY_IN_PROGRESS,
    AUTO_RECOVERY_SUCCEEDED,
    AUTO_RECOVERY_FAILED,
)

# stranded_in_ready classifier outcomes (Product contract §3).
CLASSIFICATION_LEGITIMATELY_QUEUED = "LEGITIMATELY_QUEUED"
CLASSIFICATION_READY_TOO_LONG_UNEXPLAINED = "READY_TOO_LONG_UNEXPLAINED"
CLASSIFICATION_NO_COMPATIBLE_WORKER = "NO_COMPATIBLE_WORKER"
CLASSIFICATION_DISPATCHER_UNHEALTHY = "DISPATCHER_UNHEALTHY"
CLASSIFICATION_PROFILE_CAPACITY_SATURATED = "PROFILE_CAPACITY_SATURATED"
STRANDED_CLASSIFICATIONS = (
    CLASSIFICATION_LEGITIMATELY_QUEUED,
    CLASSIFICATION_READY_TOO_LONG_UNEXPLAINED,
    CLASSIFICATION_NO_COMPATIBLE_WORKER,
    CLASSIFICATION_DISPATCHER_UNHEALTHY,
    CLASSIFICATION_PROFILE_CAPACITY_SATURATED,
)

# Recovery lifecycle events this engine recognises (read-only markers).
# Bounded + non-overlapping with the stall watchdog semantics (t_60c940e3
# lane owns execution); when the merged watchdog emits these event kinds
# the engine picks them up automatically.
EVENT_RECOVERY_STARTED = "recovery_started"
EVENT_RECOVERING = "recovering"
EVENT_RECOVERY_SUCCEEDED = "recovery_succeeded"
EVENT_RECOVERY_FAILED = "recovery_failed"
_RECOVERY_START_KINDS = {EVENT_RECOVERY_STARTED, EVENT_RECOVERING}
_RECOVERY_TERMINAL_KINDS = {EVENT_RECOVERY_SUCCEEDED, EVENT_RECOVERY_FAILED}


def attention_at_or_above(attention: Optional[str], threshold: Optional[str]) -> bool:
    """Return True when ``attention`` meets or exceeds ``threshold``."""
    if threshold is None:
        return True
    if attention not in ATTENTION_ORDER or threshold not in ATTENTION_ORDER:
        return False
    return ATTENTION_ORDER.index(attention) >= ATTENTION_ORDER.index(threshold)


def owner_action_for_attention(attention: Optional[str]) -> str:
    """Derive the owner-action field from the attention level.

    REQUIRED exactly when attention is ACTION_REQUIRED or CRITICAL; every
    other level is NONE (recommendations travel as suggested actions).
    """
    if attention in (ATTENTION_ACTION_REQUIRED, ATTENTION_CRITICAL):
        return OWNER_ACTION_REQUIRED
    return OWNER_ACTION_NONE


def attention_banner_policy(
    *,
    attention: str,
    auto_recovery_state: str = AUTO_RECOVERY_NONE,
    abnormal: bool = True,
    auto_recoverable: bool = False,
) -> bool:
    """Attention-banner decision (Product contract §5) — the ONLY place the
    ``attention_banner`` flag is decided; surfaces filter/render it, they do
    not re-derive it.

    A diagnostic lands in the operator attention banner iff:
      * its auto-recovery FAILED, or
      * its attention is CRITICAL or ACTION_REQUIRED, or
      * it is an abnormal WARNING that Hermes cannot auto-recover.
    Healthy queue/capacity conditions (INFO/NONE) never banner.
    """
    if auto_recovery_state == AUTO_RECOVERY_FAILED:
        return True
    if attention == ATTENTION_CRITICAL:
        return True
    if owner_action_for_attention(attention) == OWNER_ACTION_REQUIRED:
        return True
    if attention == ATTENTION_WARNING and abnormal and not auto_recoverable:
        return True
    return False


@dataclass
class DiagnosticAction:
    """A single recovery action attached to a diagnostic.

    The ``kind`` determines how both the UI and CLI render it:

    * ``reclaim`` / ``reassign`` — POST to the matching /tasks/:id/*
      endpoint; dashboard wires into the existing recovery popover.
    * ``unblock`` — PATCH status back to ``ready`` (for stuck-blocked
      diagnostics).
    * ``comment`` — nudge the operator to add a comment (for
      stuck-blocked tasks that need human input).
    * ``run_diagnostics`` — run a read-only diagnostics pass (board /
      dispatcher) — the primary replacement for the old opaque
      ``cli_hint`` "Check dispatcher status".
    * ``view_worker`` / ``view_queue`` — read-only navigation to the
      active worker log / the profile's ready queue.
    * ``open_docs`` — deep-link to the docs URL named in ``payload.url``.
    * ``cli_hint`` — print/copy a shell command (e.g.
      ``hermes -p <profile> auth``). No HTTP side effect; surfaces render
      it as a SECONDARY affordance only (discreet copy icon), never as the
      primary button.

    ``suggested=True`` marks the action as the recommended first step;
    the UI highlights it. Multiple actions can be suggested if they're
    equally valid.
    """

    kind: str
    label: str
    payload: dict = field(default_factory=dict)
    suggested: bool = False

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "payload": self.payload,
            "suggested": self.suggested,
        }


@dataclass
class Diagnostic:
    """One active distress signal on a task."""

    kind: str
    severity: str  # "warning" | "error" | "critical"
    title: str
    detail: str
    actions: list[DiagnosticAction] = field(default_factory=list)
    first_seen_at: int = 0
    last_seen_at: int = 0
    count: int = 1
    # Optional: the run id this diagnostic is scoped to. None = task-wide.
    run_id: Optional[int] = None
    # Optional structured payload for the UI (phantom ids, failure count).
    data: dict = field(default_factory=dict)
    # --- Operator attention axis (additive over ``severity``). Decided by
    # the rule that emits the diagnostic, finalised in
    # :func:`_finalize_operator_fields`; surfaces render, never re-derive. ---
    attention: str = ATTENTION_INFO
    # owner_action is derived from attention (NONE|REQUIRED) in the finalizer.
    owner_action: str = OWNER_ACTION_NONE
    # Short FR sentence: what Hermes does automatically about this state.
    system_action: str = ""
    # Whether the diagnostic counts toward the operator attention banner.
    # None = resolve via :func:`attention_banner_policy` in the finalizer.
    attention_banner: Optional[bool] = None
    auto_recovery_state: str = AUTO_RECOVERY_NONE
    # stranded_in_ready classifier outcome (one of STRANDED_CLASSIFICATIONS).
    classification: Optional[str] = None
    # --- Operator message (FR copy; technical details stay in title/detail/
    # data, always EN). Section labels rendered by surfaces stay EN:
    # STATUS / CAUSE / IMPACT / OWNER ACTION / SYSTEM ACTION. ---
    operator_status: str = ""
    operator_cause: str = ""
    operator_impact: str = ""
    # Replaces operator_impact for concurrency kinds (STATUS/CAUSE/RISK/…).
    operator_risk: str = ""

    def to_dict(self) -> dict:
        banner = self.attention_banner
        if banner is None:
            banner = attention_banner_policy(
                attention=self.attention,
                auto_recovery_state=self.auto_recovery_state,
            )
        return {
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "actions": [a.to_dict() for a in self.actions],
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "count": self.count,
            "run_id": self.run_id,
            "data": self.data,
            # Operator attention axis
            "attention": self.attention,
            "owner_action": owner_action_for_attention(self.attention),
            "system_action": self.system_action,
            "attention_banner": banner,
            "auto_recovery_state": self.auto_recovery_state,
            "classification": self.classification,
            "operator_status": self.operator_status,
            "operator_cause": self.operator_cause,
            "operator_impact": self.operator_impact,
            "operator_risk": self.operator_risk,
        }


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------

def _task_field(task, name, default=None):
    """Read a field from a task regardless of representation.

    Callers pass sqlite3.Row (dict-like with [] but no attribute
    access), kanban_db.Task dataclasses (attribute access), or plain
    dicts (both). This normalises them so rule functions don't have
    to branch on type each time.
    """
    if task is None:
        return default
    # sqlite Row + plain dicts both support mapping access; Row also
    # supports .keys().
    try:
        # Row raises IndexError if the key isn't a column in the query;
        # dicts return default via .get. Handle both.
        if hasattr(task, "keys") and name in task.keys():
            return task[name]
    except Exception:
        pass
    if isinstance(task, dict):
        return task.get(name, default)
    return getattr(task, name, default)


def _parse_payload(ev) -> dict:
    """Tolerate event.payload being either a dict or a JSON string."""
    p = _task_field(ev, "payload", None)
    if p is None:
        return {}
    if isinstance(p, dict):
        return p
    if isinstance(p, str):
        try:
            return json.loads(p) or {}
        except Exception:
            return {}
    return {}


def _event_kind(ev) -> str:
    return _task_field(ev, "kind", "") or ""


def _event_ts(ev) -> int:
    t = _task_field(ev, "created_at", 0)
    return int(t or 0)


def _active_hallucination_events(
    events: Iterable[Any],
    kind: str,
) -> list[Any]:
    """Return events of ``kind`` that have no ``completed``/``edited``
    event *strictly after* them. Walks chronologically: each clean
    event resets the accumulator; each matching event gets appended.

    Events must be sorted by id (i.e. arrival order); callers pass the
    task's full event list which the DB already returns in that order.
    """
    # Events arrive sorted by id asc (chronological). Walk once, track
    # which hallucination events are still "active" (no clean event
    # supersedes them).
    active: list[Any] = []
    for ev in events:
        k = _event_kind(ev)
        if k in {"completed", "edited"}:
            active.clear()
        elif k == kind:
            active.append(ev)
    return active
# Standard always-available actions. Every diagnostic can offer these as
# fallbacks regardless of kind — they're the two baseline recovery
# primitives the kernel supports.
def _generic_recovery_actions(task: Any, *, running: bool) -> list[DiagnosticAction]:
    out: list[DiagnosticAction] = []
    if running:
        out.append(DiagnosticAction(
            kind="reclaim",
            label="Reclaim task",
            payload={},
        ))
    out.append(DiagnosticAction(
        kind="reassign",
        label="Reassign to different profile",
        payload={"reclaim_first": running},
    ))
    return out


# ---------------------------------------------------------------------------
# Operator-axis helpers (FR operator copy; canonical technical terms stay EN)
# ---------------------------------------------------------------------------


def _fmt_age(age_seconds: float) -> str:
    """Render a duration in an operator-readable unit (45m / 2h / 1j)."""
    if age_seconds >= 86400:
        return f"{age_seconds / 86400:.1f}j"
    if age_seconds >= 3600:
        return f"{age_seconds / 3600:.1f}h"
    return f"{int(age_seconds / 60)}m"


def _op(
    *,
    attention: str = ATTENTION_INFO,
    system_action: str = "",
    status: str = "",
    cause: str = "",
    impact: str = "",
    risk: str = "",
    classification: Optional[str] = None,
    auto_recovery_state: str = AUTO_RECOVERY_NONE,
) -> dict:
    """Keyword bundle for the operator fields of a Diagnostic constructor.

    ``owner_action`` is derived from ``attention`` by the finalizer, so rules
    never set it by hand. ``attention_banner`` is resolved by the policy in
    the finalizer unless a rule passes it explicitly.
    """
    return {
        "attention": attention,
        "owner_action": owner_action_for_attention(attention),
        "system_action": system_action,
        "auto_recovery_state": auto_recovery_state,
        "classification": classification,
        "operator_status": status,
        "operator_cause": cause,
        "operator_impact": impact,
        "operator_risk": risk,
    }


def _run_diagnostics_action(*, suggested: bool = False) -> DiagnosticAction:
    """Primary action replacing the opaque \"Check dispatcher status\" CLI
    hint: a real read-only diagnostics run (board-level)."""
    return DiagnosticAction(
        kind="run_diagnostics",
        label="Diagnostiquer le dispatcher",
        payload={"command": "hermes kanban diagnostics"},
        suggested=suggested,
    )


def _view_worker_action(worker_id: Optional[str] = None) -> DiagnosticAction:
    payload = {}
    if worker_id:
        payload["worker_id"] = worker_id
    return DiagnosticAction(
        kind="view_worker",
        label="Voir le worker actif",
        payload=payload,
    )


def _view_queue_action(profile: str) -> DiagnosticAction:
    return DiagnosticAction(
        kind="view_queue",
        label=f"Voir la file {profile}",
        payload={"profile": profile},
    )


def _secondary_cli_hint(command: str, label: Optional[str] = None) -> DiagnosticAction:
    """Raw CLI command affordance — SECONDARY only (never suggested)."""
    return DiagnosticAction(
        kind="cli_hint",
        label=label or f"Copier la commande : {command}",
        payload={"command": command},
        suggested=False,
    )


def _task_scope_key(task: Any) -> Optional[tuple]:
    """Stable identity of the repo+branch a task writes to.

    Returns ``None`` when the task has no scoped workspace, so concurrency
    detection never groups unscoped tasks together.
    """
    ws = str(_task_field(task, "workspace_path") or "").strip()
    branch = str(_task_field(task, "branch_name") or "").strip()
    if not ws and not branch:
        return None
    project = str(_task_field(task, "project_id") or "").strip()
    # project_id is the "scope" discriminator when present; two tasks in the
    # same repo on the same branch under different projects are not dupes.
    return (ws or None, branch or None, project or None)


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

# Each rule takes (task, events, runs, now_ts, config) and returns
# zero or more Diagnostic instances. ``events`` / ``runs`` are lists of
# kanban_db.Event / kanban_db.Run (or plain dicts matching the same
# shape — for test convenience).

RuleFn = Callable[[Any, list[Any], list[Any], int, dict], list[Diagnostic]]


def _aux_slot_explicit(slot: Any) -> bool:
    """Return True if the auxiliary slot has user-supplied non-default fields.

    Defaults from ``DEFAULT_CONFIG`` use ``provider: "auto"`` with empty
    model/base_url/api_key — that path falls through to the main model. An
    "explicit" config is one where the user actively set a provider (not
    "auto"), or supplied a model / base_url / api_key.
    """
    if not isinstance(slot, dict):
        return False
    provider = str(slot.get("provider") or "").strip().lower()
    if provider and provider != "auto":
        return True
    for key in ("model", "base_url", "api_key"):
        if str(slot.get(key) or "").strip():
            return True
    return False


def _main_model_visible(raw_config: Any) -> bool:
    """Best-effort check that a main model is configured.

    Diagnostics runs in the dashboard process which may not share the CLI's
    runtime state, so we read the raw config dict. If we cannot prove the
    main model is set, we err on the side of NOT firing the diagnostic.
    """
    if not isinstance(raw_config, dict):
        return False
    model_cfg = raw_config.get("model")
    if isinstance(model_cfg, dict):
        provider = str(model_cfg.get("provider") or "").strip()
        model = str(
            model_cfg.get("default")
            or model_cfg.get("model")
            or model_cfg.get("name")
            or ""
        ).strip()
        return bool(provider and model)
    return bool(str(model_cfg or "").strip())


def triage_aux_status(config: Optional[dict]) -> Optional[dict]:
    """Inspect raw config and report whether triage paths look configured.

    Returns ``None`` when config context is unavailable (suppress diagnostic
    to avoid noisy false positives in tests / low-level callers). Otherwise
    returns a dict with:

      - ``auto_decompose``: bool — whether the dispatcher auto-runs decompose
      - ``decomposer_explicit``: bool — user-supplied decomposer slot
      - ``specifier_explicit``: bool — user-supplied specifier slot
      - ``main_model_visible``: bool — main model can serve as auto fallback
    """
    if not isinstance(config, dict):
        return None

    explicit = config.get("triage_aux_status")
    if isinstance(explicit, dict):
        return explicit

    aux = config.get("auxiliary")
    kanban_cfg = config.get("kanban") if isinstance(config.get("kanban"), dict) else {}

    # Have we been handed any config context at all? When neither auxiliary
    # nor kanban nor model keys are present, the caller is a low-level test
    # passing {} — stay silent.
    if (
        not isinstance(aux, dict)
        and not kanban_cfg
        and "model" not in config
    ):
        return None

    decomposer_explicit = False
    specifier_explicit = False
    if isinstance(aux, dict):
        decomposer_explicit = _aux_slot_explicit(aux.get("kanban_decomposer"))
        specifier_explicit = _aux_slot_explicit(aux.get("triage_specifier"))

    # ``auto_decompose`` defaults to True per kanban DEFAULT_CONFIG.
    auto_decompose = True
    if isinstance(kanban_cfg, dict) and "auto_decompose" in kanban_cfg:
        auto_decompose = bool(kanban_cfg.get("auto_decompose"))

    return {
        "auto_decompose": auto_decompose,
        "decomposer_explicit": decomposer_explicit,
        "specifier_explicit": specifier_explicit,
        "main_model_visible": _main_model_visible(config),
    }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _rule_hallucinated_cards(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Blocked-hallucination gate fires: a worker called kanban_complete
    with created_cards that didn't exist or weren't created by the
    completing profile. Task stayed in its prior state; the operator
    needs to decide how to proceed.

    Auto-clears when a successful completion (or edit) follows the
    blocked event.
    """
    hits = _active_hallucination_events(events, "completion_blocked_hallucination")
    if not hits:
        return []
    phantom_ids: list[str] = []
    first = _event_ts(hits[0])
    last = _event_ts(hits[-1])
    for ev in hits:
        payload = _parse_payload(ev)
        for pid in payload.get("phantom_cards", []) or []:
            if pid not in phantom_ids:
                phantom_ids.append(pid)
    running = _task_field(task, "status") == "running"
    actions: list[DiagnosticAction] = []
    actions.append(DiagnosticAction(
        kind="comment",
        label="Add a comment explaining what to do",
        suggested=False,
    ))
    actions.extend(_generic_recovery_actions(task, running=running))
    return [Diagnostic(
        kind="hallucinated_cards",
        severity="error",
        title="Worker claimed cards that don't exist",
        detail=(
            "The completing worker declared created_cards that either didn't "
            "exist or weren't created by its profile. The completion was "
            "blocked and the task stayed in its prior state. "
            "Usually means the worker hallucinated ids instead of capturing "
            "return values from kanban_create."
        ),
        actions=actions,
        first_seen_at=first,
        last_seen_at=last,
        count=len(hits),
        data={"phantom_ids": phantom_ids},
    )]


def _rule_triage_aux_unavailable(task, events, runs, now, cfg) -> list[Diagnostic]:
    """A triage task cannot leave triage without an auxiliary helper.

    With the auto-decompose dispatcher (kanban.auto_decompose, default True),
    triage tasks fan out via ``auxiliary.kanban_decomposer`` and fall back to
    ``auxiliary.triage_specifier`` when the decomposer returns ``fanout=false``.
    With auto-decompose off, the user must run ``hermes kanban specify``,
    which only needs ``auxiliary.triage_specifier``.

    The default slot is ``provider: auto`` → auto-falls back to the main model,
    so this rule only fires when:

      - the relevant slot is explicitly set to something broken, OR
      - the auto fallback has no main model to fall back to.

    Config context is required; pass {} from tests to keep the rule silent.
    """
    if _task_field(task, "status") != "triage":
        return []

    status = triage_aux_status(cfg)
    if status is None:
        return []

    auto_decompose = bool(status.get("auto_decompose"))
    decomposer_explicit = bool(status.get("decomposer_explicit"))
    specifier_explicit = bool(status.get("specifier_explicit"))
    main_visible = bool(status.get("main_model_visible"))

    # Determine the primary slot and whether it is usable.
    if auto_decompose:
        primary_slot = "auxiliary.kanban_decomposer"
        primary_explicit = decomposer_explicit
        fallback_slot = "auxiliary.triage_specifier"
        fallback_explicit = specifier_explicit
        primary_desc = "decomposer"
        detail_path = (
            "Auto-decompose is on, so the dispatcher needs "
            "auxiliary.kanban_decomposer (with auxiliary.triage_specifier as "
            "a fallback for non-fan-out tasks)."
        )
    else:
        primary_slot = "auxiliary.triage_specifier"
        primary_explicit = specifier_explicit
        fallback_slot = "auxiliary.kanban_decomposer"
        fallback_explicit = decomposer_explicit
        primary_desc = "specifier"
        detail_path = (
            "Auto-decompose is off, so triage tasks need "
            "`hermes kanban specify`, which uses auxiliary.triage_specifier."
        )

    # The primary slot is usable when either: it was explicitly configured by
    # the user, OR the default `provider: auto` can fall back to the main
    # model. If both fail, we have a real configuration gap.
    if primary_explicit or main_visible:
        return []

    task_id = _task_field(task, "id") or "<task_id>"
    actions = [
        DiagnosticAction(
            kind="cli_hint",
            label=f"Configure {primary_slot}",
            payload={
                "command": (
                    f"hermes config set {primary_slot}.provider auto"
                )
            },
            suggested=True,
        ),
    ]
    if not fallback_explicit and not main_visible:
        actions.append(DiagnosticAction(
            kind="cli_hint",
            label=f"Or configure fallback {fallback_slot}",
            payload={
                "command": (
                    f"hermes config set {fallback_slot}.provider auto"
                )
            },
        ))
    if not auto_decompose:
        actions.append(DiagnosticAction(
            kind="cli_hint",
            label=f"Specify manually: hermes kanban specify {task_id}",
            payload={"command": f"hermes kanban specify {task_id}"},
        ))

    return [Diagnostic(
        kind="triage_aux_unavailable",
        severity="warning",
        title=f"Triage {primary_desc} has no usable model",
        detail=(
            f"This task is still in triage and no working auxiliary model is "
            f"visible to the dispatcher. {detail_path} The default slot uses "
            f"`provider: auto` which falls back to the main model, but no main "
            f"model is configured either. Configure the slot directly or set a "
            f"main model so the auto fallback can take over."
        ),
        actions=actions,
        first_seen_at=now,
        last_seen_at=now,
        count=1,
        data={
            "task_id": task_id,
            "auto_decompose": auto_decompose,
            "primary_slot": primary_slot,
            "main_model_visible": main_visible,
        },
    )]


def _rule_prose_phantom_refs(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Advisory prose-scan: the completion summary mentions ``t_<hex>``
    ids that don't resolve. Non-blocking; surfaced as a warning only.

    Auto-clears when a fresh clean completion arrives AFTER the
    suspected event.
    """
    hits = _active_hallucination_events(events, "suspected_hallucinated_references")
    if not hits:
        return []
    phantom_refs: list[str] = []
    for ev in hits:
        for pid in _parse_payload(ev).get("phantom_refs", []) or []:
            if pid not in phantom_refs:
                phantom_refs.append(pid)
    running = _task_field(task, "status") == "running"
    return [Diagnostic(
        kind="prose_phantom_refs",
        severity="warning",
        title="Completion summary references unknown task ids",
        detail=(
            "The completion summary mentions task ids that don't resolve "
            "in this board's database. The completion itself succeeded, "
            "but downstream consumers parsing the summary may be pointed "
            "at cards that never existed."
        ),
        actions=_generic_recovery_actions(task, running=running),
        first_seen_at=_event_ts(hits[0]),
        last_seen_at=_event_ts(hits[-1]),
        count=len(hits),
        data={"phantom_refs": phantom_refs},
    )]


def _rule_repeated_failures(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Task's unified ``consecutive_failures`` counter is climbing —
    something about this task+profile combo is broken and each retry
    fails the same way. Triggers regardless of the specific failure
    mode (spawn error, timeout, crash) because operationally they
    all look the same: the kernel keeps retrying and the operator
    needs to intervene.

    Threshold: cfg["failure_threshold"]. Runtime callers should derive
    this from ``kanban.failure_limit`` unless the user explicitly set a
    diagnostics threshold, so the signal does not lag behind the
    dispatcher's circuit breaker.

    Accepts the legacy ``spawn_failure_threshold`` config key for
    back-compat.

    Terminal statuses are exempt: a done/archived card has nothing left
    to retry, so a lingering failure streak is history, not a signal.
    (``complete_task`` resets the counter, but a manual done — e.g. a
    dashboard drag — ends no run and used to leave the flag stuck.)

    A fresh attempt in flight (``running``) is also exempt: retrying a
    task should clear the stale failure banner until this attempt also
    resolves. Otherwise a card that's actively trying again still shows
    "failed Nx", which reads as a current failure. It re-fires if the new
    run fails too (status leaves ``running`` with a recorded outcome).
    """
    if _task_field(task, "status") in ("done", "archived", "running"):
        return []
    threshold = _positive_int(cfg.get(
        "failure_threshold",
        cfg.get("spawn_failure_threshold", 3),
    ), 3)
    failure_limit = _positive_int(cfg.get("failure_limit"), threshold)
    # Read the new unified counter name, with a fallback to the legacy
    # column name so this rule keeps working against old DB rows the
    # caller somehow materialised without running the migration.
    failures = (
        _task_field(task, "consecutive_failures", None)
        if _task_field(task, "consecutive_failures", None) is not None
        else _task_field(task, "spawn_failures", 0)
    )
    if failures is None or failures < threshold:
        return []
    last_err = (
        _task_field(task, "last_failure_error", None)
        if _task_field(task, "last_failure_error", None) is not None
        else _task_field(task, "last_spawn_error", None)
    )
    assignee = _task_field(task, "assignee")

    # Classify the most recent failure by peeking at run outcomes so
    # the title + suggested action can be specific without a separate
    # per-outcome rule.
    ordered_runs = sorted(runs, key=lambda r: _task_field(r, "id", 0))
    most_recent_outcome = None
    for r in reversed(ordered_runs):
        oc = _task_field(r, "outcome")
        if oc in {"spawn_failed", "timed_out", "crashed"}:
            most_recent_outcome = oc
            break

    actions: list[DiagnosticAction] = []
    if most_recent_outcome == "spawn_failed" and assignee and assignee != "default":
        # Spawn is failing specifically — profile setup issue.
        actions.append(DiagnosticAction(
            kind="cli_hint",
            label=f"Verify profile: hermes -p {assignee} doctor",
            payload={"command": f"hermes -p {assignee} doctor"},
            suggested=True,
        ))
        actions.append(DiagnosticAction(
            kind="cli_hint",
            label=f"Fix profile auth: hermes -p {assignee} auth",
            payload={"command": f"hermes -p {assignee} auth"},
        ))
    elif most_recent_outcome in {"timed_out", "crashed"}:
        # Worker got off the ground but died. Logs are the right place
        # to diagnose; reclaim/reassign are the recovery levers.
        task_id = _task_field(task, "id")
        if task_id:
            actions.append(DiagnosticAction(
                kind="cli_hint",
                label=f"Check logs: hermes kanban log {task_id}",
                payload={"command": f"hermes kanban log {task_id}"},
                suggested=True,
            ))
    actions.extend(_generic_recovery_actions(
        task, running=_task_field(task, "status") == "running",
    ))

    severity = "critical" if failures >= threshold * 2 else "error"
    err_text = (last_err or "").strip() if last_err else ""
    err_snippet = err_text[:500] + ("…" if len(err_text) > 500 else "") if err_text else ""
    outcome_label = {
        "spawn_failed": "spawn",
        "timed_out": "timeout",
        "crashed": "crash",
    }.get(most_recent_outcome or "", "failure")
    if err_snippet:
        title = f"Agent {outcome_label} x{failures}: {err_snippet.splitlines()[0][:160]}"
        detail = (
            f"This task has failed {failures} times in a row "
            f"(most recent: {outcome_label}). Full last error:\n\n"
            f"{err_snippet}\n\n"
            f"The dispatcher circuit breaker is configured for "
            f"{failure_limit} consecutive non-success attempts. Fix the "
            f"root cause and reclaim or unblock the task to retry."
        )
    else:
        title = f"Agent {outcome_label} x{failures} (no error recorded)"
        detail = (
            f"This task has failed {failures} times in a row "
            f"(most recent: {outcome_label}) but no error text was "
            f"captured. Check the suggested command or the worker log."
        )
    return [Diagnostic(
        kind="repeated_failures",
        severity=severity,
        title=title,
        detail=detail,
        actions=actions,
        first_seen_at=now,
        last_seen_at=now,
        count=failures,
        data={
            "consecutive_failures": failures,
            "most_recent_outcome": most_recent_outcome,
            "last_error": last_err,
            "failure_threshold": threshold,
            "failure_limit": failure_limit,
        },
    )]


def _rule_repeated_crashes(task, events, runs, now, cfg) -> list[Diagnostic]:
    """The worker spawns fine but keeps crashing mid-run. Check the last
    N runs' outcomes; N consecutive ``crashed`` without a successful
    ``completed`` means something about the task + profile combo is
    broken (OOM, missing dependency, tool it needs is down).

    Threshold: cfg["crash_threshold"] (default 2).

    Narrower than ``repeated_failures`` — fires earlier (2 crashes vs 3
    total failures) so the operator gets a crash-specific heads-up
    before the unified rule kicks in. Suppresses itself when the
    unified rule is also about to fire, to avoid double-flagging.

    Terminal statuses are exempt for the same reason as
    ``repeated_failures`` — with one extra wrinkle: this rule reads run
    history, and a manual done (dashboard drag) appends no ``completed``
    run to break the crash streak, so the flag was permanent (#kanban
    desktop dogfood). Done means done.

    ``running`` is exempt too: a fresh attempt is in flight, and its
    in-flight run (no outcome yet) doesn't break the trailing crash scan,
    so a retried card kept showing "crashed Nx" over an active run. The
    banner re-fires if the new attempt also crashes.
    """
    if _task_field(task, "status") in ("done", "archived", "running"):
        return []
    failure_threshold = int(cfg.get(
        "failure_threshold",
        cfg.get("spawn_failure_threshold", 3),
    ))
    unified_counter = (
        _task_field(task, "consecutive_failures", 0) or 0
    )
    # Unified rule will catch this — let it handle to avoid double fire.
    if unified_counter >= failure_threshold:
        return []

    threshold = int(cfg.get("crash_threshold", 2))
    ordered = sorted(runs, key=lambda r: _task_field(r, "id", 0))
    # Count trailing consecutive 'crashed' outcomes.
    consecutive = 0
    last_err = None
    for r in reversed(ordered):
        outcome = _task_field(r, "outcome")
        if outcome == "crashed":
            consecutive += 1
            if last_err is None:
                last_err = _task_field(r, "error")
        elif outcome in {"completed", "reclaimed"}:
            # A success (or manual reclaim) breaks the streak.
            break
        else:
            # Other outcomes (timed_out, blocked, spawn_failed, gave_up)
            # aren't crash signals — don't count them, but they also
            # don't break the crash streak.
            continue
    if consecutive < threshold:
        return []
    task_id = _task_field(task, "id")
    actions: list[DiagnosticAction] = []
    if task_id:
        actions.append(DiagnosticAction(
            kind="cli_hint",
            label=f"Check logs: hermes kanban log {task_id}",
            payload={"command": f"hermes kanban log {task_id}"},
            suggested=True,
        ))
    running = _task_field(task, "status") == "running"
    actions.extend(_generic_recovery_actions(task, running=running))
    severity = "critical" if consecutive >= threshold * 2 else "error"
    # Put the actual error up-front so operators see WHAT broke without
    # having to open the logs. Truncate defensively — these can be huge
    # (full tracebacks).
    err_text = (last_err or "").strip() if last_err else ""
    err_snippet = err_text[:500] + ("…" if len(err_text) > 500 else "") if err_text else ""
    if err_snippet:
        title = f"Agent crashed {consecutive}x: {err_snippet.splitlines()[0][:160]}"
        detail = (
            f"The last {consecutive} runs ended with outcome=crashed. "
            f"Full last error:\n\n{err_snippet}"
        )
    else:
        title = f"Agent crashed {consecutive}x (no error recorded)"
        detail = (
            f"The last {consecutive} runs ended with outcome=crashed but "
            f"no error text was captured. Check the worker log for more."
        )
    return [Diagnostic(
        kind="repeated_crashes",
        severity=severity,
        title=title,
        detail=detail,
        actions=actions,
        first_seen_at=now,
        last_seen_at=now,
        count=consecutive,
        data={"consecutive_crashes": consecutive, "last_error": last_err},
    )]


def _rule_review_dependency_deadlock(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Detect a legacy review handoff that starves downstream children.

    Older workers were instructed to sticky-block an implementation with a
    ``review-required:`` reason. A separately modelled reviewer child cannot
    promote until that parent is terminal, so the lane has no autonomous next
    step. This compatibility diagnostic is graph-aware but deliberately leaves
    both the dependency graph and the user's sticky block unchanged.
    """
    if _task_field(task, "status") != "blocked":
        return []

    latest_block = None
    for event in events:
        if _event_kind(event) == "blocked":
            latest_block = event
    if latest_block is None:
        return []
    reason = str(_parse_payload(latest_block).get("reason") or "").strip()
    if not reason.lower().startswith("review-required:"):
        return []

    graph = cfg.get("_graph")
    if not isinstance(graph, dict):
        return []
    waiting_children = [
        child
        for child in (graph.get("children") or [])
        if isinstance(child, dict) and child.get("status") == "todo"
    ]
    if not waiting_children:
        return []

    task_id = str(_task_field(task, "id") or "")
    child_ids = [
        str(child.get("id"))
        for child in waiting_children
        if child.get("id")
    ]
    actions: list[DiagnosticAction] = []
    if task_id:
        actions.append(DiagnosticAction(
            kind="cli_hint",
            label="Complete the finished implementation phase",
            payload={"command": f"hermes kanban complete {task_id}"},
            suggested=True,
        ))
    if task_id and child_ids:
        actions.append(DiagnosticAction(
            kind="cli_hint",
            label="Or unlink the incorrectly gated reviewer",
            payload={"command": f"hermes kanban unlink {task_id} {child_ids[0]}"},
        ))

    blocked_at = _event_ts(latest_block) or now
    return [Diagnostic(
        kind="review_dependency_deadlock",
        severity="error",
        title=f"Review handoff blocks {len(child_ids)} dependent task(s)",
        detail=(
            "This implementation is sticky-blocked for review while its "
            "downstream task(s) require the implementation to be done or "
            "archived before they can run. Complete the finished phase, unlink "
            "the incorrect dependency, or migrate this workflow to the "
            "first-class review lifecycle."
        ),
        actions=actions,
        first_seen_at=blocked_at,
        last_seen_at=blocked_at,
        count=len(child_ids),
        data={
            "blocked_parent_id": task_id,
            "waiting_child_ids": child_ids,
            "block_reason": reason,
        },
    )]


def _rule_stuck_in_blocked(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Task has been in ``blocked`` status for too long without a comment.

    Threshold: cfg["blocked_stale_hours"] (default 24).

    Approval fast-path (Operator Diagnostics Clarity case E): when the most
    recent block explicitly carries an approval decision (payload
    ``decision_class=APPROVAL_REQUIRED`` or a reason mentioning "approval"),
    the diagnostic fires IMMEDIATELY at ACTION_REQUIRED — a task awaiting an
    owner decision must not sit silently for 24h.

    Surfaced as a warning so humans know there's a pending unblock.
    """
    hours = float(cfg.get("blocked_stale_hours", 24))
    status = _task_field(task, "status")
    if status != "blocked":
        return []
    # Find the most recent ``blocked`` event.
    last_blocked_ts = 0
    latest_block = None
    for ev in events:
        if _event_kind(ev) == "blocked":
            t = _event_ts(ev)
            if t >= last_blocked_ts:
                last_blocked_ts = t
                latest_block = ev
    if last_blocked_ts == 0:
        return []
    age_hours = (now - last_blocked_ts) / 3600.0

    block_payload = _parse_payload(latest_block)
    reason = str(block_payload.get("reason") or "").strip()
    decision_class = str(block_payload.get("decision_class") or "").strip()
    approval_block = (
        decision_class.upper() == "APPROVAL_REQUIRED"
        or "approval" in reason.lower()
    )
    if not approval_block and age_hours < hours:
        return []
    # Any comment / unblock after the block breaks the "stale" signal —
    # except an explicit approval block, which stays live until answered.
    for ev in events:
        if (
            not approval_block
            and _event_kind(ev) in {"commented", "unblocked"}
            and _event_ts(ev) > last_blocked_ts
        ):
            return []

    task_id = str(_task_field(task, "id") or "")
    actions: list[DiagnosticAction] = []
    if approval_block:
        actions.append(DiagnosticAction(
            kind="comment",
            label="Répondre à la demande d'approbation",
            suggested=True,
        ))
        actions.append(DiagnosticAction(
            kind="unblock",
            label="Débloquer la tâche",
            payload={},
        ))
    else:
        actions.append(DiagnosticAction(
            kind="comment",
            label="Add a comment / unblock the task",
            suggested=True,
        ))

    data: dict = {"blocked_at": last_blocked_ts, "age_hours": round(age_hours, 1)}
    if approval_block:
        why = reason or (
            "Décision APPROVAL_REQUIRED (déterministe, irréversible ou hors "
            "périmètre AUTO)."
        )
        data.update({
            "decision_class": "APPROVAL_REQUIRED",
            "required_action": f"Approuver la suite de la tâche {task_id}",
            "why": why,
        })
    title = (
        "En attente de votre approbation — aucune progression automatique possible"
        if approval_block else
        f"Task has been blocked for {int(age_hours)}h"
    )
    op_status = (
        "En attente de votre approbation — aucune progression automatique possible."
        if approval_block else
        "Tâche bloquée sans échange récent."
    )
    op_cause = (
        f"Décision {decision_class or 'APPROVAL_REQUIRED'} demandée : {reason or 'approbation requise'}."
        if approval_block else
        "Aucun commentaire ni tentative d'unblock depuis le passage en blocked."
    )
    op_impact = "La suite du workflow est suspendue."
    op_system = (
        "Aucune — le workflow attend votre décision explicite."
        if approval_block else
        "Aucune — une tâche bloquée attend une décision humaine."
    )
    diag = Diagnostic(
        kind="stuck_in_blocked",
        severity="warning",
        title=title,
        detail=(
            "This task is blocked awaiting an explicit operator approval "
            "(decision_class=APPROVAL_REQUIRED)."
            if approval_block else
            f"This task transitioned to blocked {int(age_hours)}h ago and "
            f"has had no comments or unblock attempts since. Blocked tasks "
            f"are waiting for human input — check the block reason and "
            f"either unblock with feedback or answer with a comment."
        ),
        actions=actions,
        first_seen_at=last_blocked_ts,
        last_seen_at=last_blocked_ts,
        count=1,
        data=data,
        **_op(
            attention=ATTENTION_ACTION_REQUIRED,
            system_action=op_system,
            status=op_status,
            cause=op_cause,
            impact=op_impact,
        ),
    )
    diag.attention_banner = True
    return [diag]


def _rule_block_unblock_cycling(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Task has cycled through blocked → unblocked many times — the
    ``unblock`` is not fixing the underlying problem and the worker
    keeps re-blocking for substantially the same reason.

    ``_rule_stuck_in_blocked`` resets its timer on any ``commented`` /
    ``unblocked`` event, so a task that cycles every few minutes is
    invisible to it regardless of how many times it cycles (#29747
    gap 1). This rule complements that one by counting block→unblock
    cycles in a sliding window.

    Threshold: cfg["block_cycle_threshold"] (default 3) cycles within
    cfg["block_cycle_window_seconds"] (default 24h).
    """
    threshold = _positive_int(cfg.get("block_cycle_threshold"), 3)
    window_seconds = float(cfg.get("block_cycle_window_seconds", 24 * 3600))
    cycle_cutoff = now - window_seconds

    # Walk events chronologically (arrival order — callers pre-sort by
    # id, which is the canonical chronological order; ``created_at``
    # alone is insufficient because multiple events can share the same
    # second).  Count "blocked after unblocked" transitions: every time
    # a blocked event follows at least one unblocked event since the
    # last cycle was counted, that's a new cycle.
    cycles = 0
    seen_unblock_since_last_cycle = False
    initial_blocked_ts = 0
    last_cycle_blocked_ts = 0
    for ev in events:
        ts = _event_ts(ev)
        if ts < cycle_cutoff:
            continue
        kind = _event_kind(ev)
        if kind == "blocked":
            if initial_blocked_ts == 0:
                initial_blocked_ts = ts
            if seen_unblock_since_last_cycle:
                cycles += 1
                last_cycle_blocked_ts = ts
                seen_unblock_since_last_cycle = False
        elif kind == "unblocked":
            seen_unblock_since_last_cycle = True

    if cycles < threshold:
        return []

    task_id = _task_field(task, "id")
    actions: list[DiagnosticAction] = []
    if task_id:
        actions.append(DiagnosticAction(
            kind="cli_hint",
            label=f"Check block reasons: hermes kanban events {task_id}",
            payload={"command": f"hermes kanban events {task_id}"},
            suggested=True,
        ))
    return [Diagnostic(
        kind="block_unblock_cycling",
        severity="warning",
        title=f"Task block→unblock cycled {cycles}x in {int(window_seconds/3600)}h",
        detail=(
            f"This task has been blocked {cycles} times after being "
            "unblocked, suggesting the unblock is not addressing the "
            "root cause and the worker keeps hitting the same wall. "
            "Review the block reasons in the event history; a different "
            "intervention (reassign, change scope, archive) may be needed."
        ),
        actions=actions,
        first_seen_at=int(initial_blocked_ts) if initial_blocked_ts else int(now),
        last_seen_at=int(last_cycle_blocked_ts) if last_cycle_blocked_ts else int(now),
        count=cycles,
        data={
            "cycles": cycles,
            "window_seconds": int(window_seconds),
        },
    )]


def _stranded_evidence(task: Any, board_context: Optional[dict], now: int) -> dict:
    """Pull the classifier's decision inputs from the read-only board context.

    Fail-safe: any input that is ABSENT stays ``None``/``False``-neutral —
    absence of evidence is never treated as health (Product contract §3.2).
    Callers build the context via :func:`build_board_context` or by hand
    (tests / Mission Control projection).
    """
    bc = board_context or {}
    assignee = str(_task_field(task, "assignee") or "").strip()
    task_id = str(_task_field(task, "id") or "")

    running_map = bc.get("running_by_assignee") or {}
    queue_map = bc.get("queue_by_assignee") or {}
    progressed_map = bc.get("queue_progressed_by_assignee") or {}
    slot_map = bc.get("expected_slot_freed_by_assignee") or {}
    attempts_map = bc.get("attempts_by_task") or {}

    running = [dict(s) for s in (running_map.get(assignee) or [])]
    queue = list(queue_map.get(assignee) or [])
    position = None
    for i, q in enumerate(queue):
        if str(q.get("id") or "") == task_id:
            position = i
            break

    # Assignee validity: positive proof comes from the roster (profiles +
    # registered lanes). Invalid is only claimed when the roster is known and
    # the assignee is in neither set.
    profiles = bc.get("profiles")
    lanes = bc.get("lanes")
    assignee_valid: Optional[bool] = None
    compatible: Optional[bool] = None
    if isinstance(profiles, list):
        if assignee in profiles:
            assignee_valid = True
            compatible = True
        else:
            lane_spawnable = lanes.get(assignee) if isinstance(lanes, dict) else None
            if lane_spawnable is not None:
                assignee_valid = True
                compatible = bool(lane_spawnable)
            elif isinstance(lanes, dict):
                assignee_valid = False
                compatible = False

    return {
        "assignee": assignee,
        "dispatcher": bc.get("dispatcher"),
        "profiles": profiles,
        "lanes": lanes,
        "profile_cap": bc.get("profile_cap"),
        "board_cap": bc.get("board_cap"),
        "assignee_valid": assignee_valid,
        "compatible_worker_available": compatible,
        "running": running,
        "queue_position": position,
        "priority": _task_field(task, "priority", 0),
        "queue_progression": bool(progressed_map.get(assignee)),
        "expected_slot_availability": bool(slot_map.get(assignee)),
        "ever_attempted": bool((attempts_map.get(task_id) or 0) > 0),
    }


def _age_technical_severity(age_seconds: float, threshold_seconds: float) -> str:
    """Age-based technical severity for stranded tasks (unchanged rungs:
    warning below 2x, error to 6x, critical beyond)."""
    if age_seconds >= threshold_seconds * 6:
        return "critical"
    if age_seconds >= threshold_seconds * 2:
        return "error"
    return "warning"


def _unexplained_outcome(evidence: dict, age_seconds: float, threshold_seconds: float) -> dict:
    """Build the conservative READY_TOO_LONG_UNEXPLAINED outcome (fail-safe).

    Attention escalates with age: WARNING below 2x threshold,
    ACTION_REQUIRED to 6x, CRITICAL beyond. Banner always on — an abnormal,
    non-auto-recoverable condition.
    """
    dispatcher = evidence["dispatcher"]
    dispatcher_healthy = (
        isinstance(dispatcher, dict) and dispatcher.get("healthy") is True
    )
    severity = _age_technical_severity(age_seconds, threshold_seconds)
    if age_seconds >= threshold_seconds * 6:
        attention = ATTENTION_CRITICAL
    elif age_seconds >= threshold_seconds * 2:
        attention = ATTENTION_ACTION_REQUIRED
    else:
        attention = ATTENTION_WARNING
    assignee = evidence["assignee"]
    cause_bits = [
        f"Assignee {assignee} valide" if evidence["assignee_valid"] is True else
        "Assignee non vérifiable",
    ]
    if dispatcher_healthy:
        cause_bits.append("dispatcher sain")
    cause_bits.append("capacité disponible et aucun claim observé dans la fenêtre")
    if evidence["expected_slot_availability"]:
        cause_bits.append("un slot s'est libéré sans que la tâche soit prise")
    cause_bits.append("aucune explication légitime (la file n'avance pas)")
    return {
        "classification": CLASSIFICATION_READY_TOO_LONG_UNEXPLAINED,
        "attention": attention,
        "banner": attention_banner_policy(attention=attention),
        "severity": severity,
        "title": f"Prête depuis {_fmt_age(age_seconds)} — aucun worker ne l'a réclamée",
        "status": (
            f"Prête depuis {_fmt_age(age_seconds)} — capacité disponible et aucun "
            "worker ne l'a réclamée."
        ),
        "cause": "; ".join(cause_bits) + ".",
        "impact": (
            "La tâche ne progresse pas alors qu'elle pourrait être exécutée."
        ),
        "system_action": "Hermes continue de proposer la tâche à chaque tick.",
        "data_extra": {
            "expected_slot_availability": evidence["expected_slot_availability"],
            "queue_position": evidence["queue_position"],
            "ever_attempted": evidence["ever_attempted"],
        },
    }


def _classify_ready_outcome(
    evidence: dict,
    age_seconds: float,
    threshold_seconds: float,
) -> dict:
    """Classify a stranded ready task into one of the 5 normative outcomes.

    Returns a dict of the outcome's operator fields:
    ``classification / attention / banner / severity / title / status /
    cause / impact / system_action``.
    """
    assignee = evidence["assignee"]
    dispatcher = evidence["dispatcher"]
    running = evidence["running"]
    cap = evidence["profile_cap"] or evidence["board_cap"]

    # P1 — dispatcher unhealth requires dispatcher EVIDENCE, never age alone.
    if isinstance(dispatcher, dict) and dispatcher.get("healthy") is False:
        board_impact = bool(dispatcher.get("board_impact"))
        attention = ATTENTION_CRITICAL if board_impact else ATTENTION_ACTION_REQUIRED
        return {
            "classification": CLASSIFICATION_DISPATCHER_UNHEALTHY,
            "attention": attention,
            "banner": True,
            "severity": "critical" if board_impact else "error",
            "title": "Dispatcher suspect — pas de tick récent",
            "status": "Dispatcher suspect — pas de tick récent.",
            "cause": (
                "Aucune activité dispatcher observée dans la fenêtre "
                "(lock/heartbeat/tick)."
            ),
            "impact": (
                "Plusieurs profiles et files sont affamés (impact board)."
                if board_impact else
                "Cette tâche et sa file ne progressent pas."
            ),
            "system_action": "Aucune — le dispatcher n'émet plus de tick.",
            "data_extra": {
                "dispatcher_board_impact": board_impact,
                "dispatcher_last_observed_ts": dispatcher.get("last_tick_ts"),
            },
        }

    # P2 — no compatible worker (proven roster mismatch).
    if evidence["assignee_valid"] is False or evidence["compatible_worker_available"] is False:
        return {
            "classification": CLASSIFICATION_NO_COMPATIBLE_WORKER,
            "attention": ATTENTION_ACTION_REQUIRED,
            "banner": True,
            "severity": "error",
            "title": f"Aucun worker compatible pour le profile {assignee}",
            "status": (
                f"Aucun worker compatible pour le profile {assignee} "
                "(assignee invalide ou lane indisponible)."
            ),
            "cause": (
                f"Le profile/lane {assignee} n'est pas dans le roster "
                "(profiles/lanes enregistrés) ou n'est pas spawnable."
            ),
            "impact": (
                "Cette tâche ne sera jamais claimée tant que l'assignee "
                "n'est pas corrigé."
            ),
            "system_action": (
                "Aucune — Hermes ne réassigne pas automatiquement sans décision."
            ),
            "data_extra": {
                "assignee_valid": evidence["assignee_valid"],
                "compatible_worker_available": evidence["compatible_worker_available"],
            },
        }

    # P3 — profile capacity saturated by running workers of the same assignee.
    if isinstance(cap, int) and cap >= 1 and len(running) >= cap:
        fresh = [s for s in running if not bool(s.get("stale"))]
        n = len(running)
        if fresh:
            # A healthy running sibling legitimately holds the slot.
            if evidence["expected_slot_availability"] and not evidence["queue_progression"]:
                # A slot freed in the window yet nobody claimed this task —
                # not a legitimate wait even with a sibling currently running.
                return _unexplained_outcome(evidence, age_seconds, threshold_seconds)
            if evidence["queue_position"] is not None and evidence["queue_position"] >= 1:
                return {
                    "classification": CLASSIFICATION_LEGITIMATELY_QUEUED,
                    "attention": ATTENTION_INFO,
                    "banner": False,
                    "severity": "warning",
                    "title": "En attente légitime de capacité worker",
                    "status": (
                        f"En attente légitime de capacité worker — profile "
                        f"{assignee} occupé par {fresh[0].get('id') or 'un worker'} "
                        f"(running), position {evidence['queue_position']}, "
                        f"priority {evidence['priority']}."
                    ),
                    "cause": (
                        f"Capacité du profile atteinte ({n}/{cap}) ; dispatcher "
                        f"sain ; la file avance."
                    ),
                    "impact": "Aucun — la tâche sera claimée dès qu'un slot se libère.",
                    "system_action": (
                        "Le dispatcher claim la tâche au prochain tick disponible."
                    ),
                    "data_extra": {
                        "profile_cap": cap,
                        "running_siblings": [s.get("id") for s in running],
                        "queue_position": evidence["queue_position"],
                        "priority": evidence["priority"],
                    },
                }
            if evidence["queue_progression"]:
                return {
                    "classification": CLASSIFICATION_LEGITIMATELY_QUEUED,
                    "attention": ATTENTION_INFO,
                    "banner": False,
                    "severity": "warning",
                    "title": "En attente légitime de capacité worker",
                    "status": (
                        f"En attente légitime de capacité worker — profile "
                        f"{assignee} occupé par {fresh[0].get('id') or 'un worker'} "
                        f"(running)."
                    ),
                    "cause": (
                        f"Capacité du profile atteinte ({n}/{cap}) ; dispatcher "
                        f"sain ; la file avance."
                    ),
                    "impact": "Aucun — la tâche sera claimée dès qu'un slot se libère.",
                    "system_action": (
                        "Le dispatcher claim la tâche au prochain tick disponible."
                    ),
                    "data_extra": {
                        "profile_cap": cap,
                        "running_siblings": [s.get("id") for s in running],
                        "queue_position": evidence["queue_position"],
                        "priority": evidence["priority"],
                    },
                }
            # Healthy sibling, but no queue-progression / position proof yet:
            # still an expected wait, informational only.
            return {
                "classification": CLASSIFICATION_PROFILE_CAPACITY_SATURATED,
                "attention": ATTENTION_INFO,
                "banner": False,
                "severity": "warning",
                "title": "Capacité worker saturée — attente normale",
                "status": (
                    f"Capacité worker saturée — profile {assignee} : "
                    f"{n}/{cap} running ; attente normale."
                ),
                "cause": (
                    f"Capacité du profile atteinte ({n}/{cap}) ; un worker actif "
                    "occupe le slot."
                ),
                "impact": "Aucun — la tâche sera claimée dès qu'un slot se libère.",
                "system_action": (
                    "Le dispatcher claim la tâche au prochain tick disponible."
                ),
                "data_extra": {
                    "profile_cap": cap,
                    "running_siblings": [s.get("id") for s in running],
                },
            }
        # No fresh sibling: every running worker of this profile is stale.
        if isinstance(dispatcher, dict) and dispatcher.get("healthy") is True:
            # The dispatcher reclaims stale claims on its next tick — expected.
            return {
                "classification": CLASSIFICATION_PROFILE_CAPACITY_SATURATED,
                "attention": ATTENTION_INFO,
                "banner": False,
                "severity": "warning",
                "title": "Capacité worker saturée — attente normale",
                "status": (
                    f"Capacité worker saturée — profile {assignee} : "
                    f"{n}/{cap} running ; attente normale."
                ),
                "cause": (
                    f"Capacité du profile atteinte ({n}/{cap}) par des workers sans "
                    "heartbeat récent ; le dispatcher les réclame au prochain tick."
                ),
                "impact": "Aucun — la tâche sera claimée après réclamation des slots.",
                "system_action": (
                    "Le dispatcher réclame les claims expirés/sans heartbeat puis "
                    "claim cette tâche."
                ),
                "data_extra": {
                    "profile_cap": cap,
                    "running_siblings": [s.get("id") for s in running],
                    "stale_siblings": True,
                },
            }
        # Stale siblings + no dispatcher-health proof: abnormal, not proven
        # auto-recoverable → surface it.
        return {
            "classification": CLASSIFICATION_PROFILE_CAPACITY_SATURATED,
            "attention": ATTENTION_WARNING,
            "banner": True,
            "severity": "warning",
            "title": "Capacité worker saturée par des workers sans heartbeat",
            "status": (
                f"Capacité worker saturée par des workers sans heartbeat récent — "
                f"profile {assignee} : {n}/{cap} running."
            ),
            "cause": (
                f"Les workers running du profile {assignee} n'ont pas de heartbeat "
                "récent (stale) et rien ne prouve que le dispatcher les réclamera."
            ),
            "impact": "Cette tâche reste en file tant que les slots ne sont pas libérés.",
            "system_action": (
                "Le dispatcher réclame normalement les claims expirés/sans heartbeat "
                "au prochain tick."
            ),
            "data_extra": {
                "profile_cap": cap,
                "running_siblings": [s.get("id") for s in running],
                "stale_siblings": True,
            },
        }

    # P4/P5 — capacity free or unknown: a slot freed and wasn't taken, or
    # nothing explains the wait. Fail-safe default (conservative):
    # READY_TOO_LONG_UNEXPLAINED warning, in the attention banner.
    dispatcher_healthy = (
        isinstance(dispatcher, dict) and dispatcher.get("healthy") is True
    )
    severity = _age_technical_severity(age_seconds, threshold_seconds)
    if age_seconds >= threshold_seconds * 6:
        attention = ATTENTION_CRITICAL
    elif age_seconds >= threshold_seconds * 2:
        attention = ATTENTION_ACTION_REQUIRED
    else:
        attention = ATTENTION_WARNING
    cause_bits = [
        f"Assignee {assignee} valide" if evidence["assignee_valid"] is True else
        "Assignee non vérifiable",
    ]
    if dispatcher_healthy:
        cause_bits.append("dispatcher sain")
    cause_bits.append("capacité disponible et aucun claim observé dans la fenêtre")
    if evidence["expected_slot_availability"]:
        cause_bits.append("un slot s'est libéré sans que la tâche soit prise")
    cause_bits.append("aucune explication légitime (la file n'avance pas)")
    return {
        "classification": CLASSIFICATION_READY_TOO_LONG_UNEXPLAINED,
        "attention": attention,
        "banner": attention_banner_policy(attention=attention),
        "severity": severity,
        "title": f"Prête depuis {_fmt_age(age_seconds)} — aucun worker ne l'a réclamée",
        "status": (
            f"Prête depuis {_fmt_age(age_seconds)} — capacité disponible et aucun "
            "worker ne l'a réclamée."
        ),
        "cause": "; ".join(cause_bits) + ".",
        "impact": (
            "La tâche ne progresse pas alors qu'elle pourrait être exécutée."
        ),
        "system_action": "Hermes continue de proposer la tâche à chaque tick.",
        "data_extra": {
            "expected_slot_availability": evidence["expected_slot_availability"],
            "queue_position": evidence["queue_position"],
            "ever_attempted": evidence["ever_attempted"],
        },
    }


def _rule_stranded_in_ready(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Task has been in ``ready`` status for too long without any worker
    claiming it — now a CLASSIFIER, not a bare age signal.

    Threshold: cfg["stranded_threshold_seconds"] (default 1800 = 30 min).

    The old identity-agnostic rule emitted one undifferentiated "no worker"
    warning for every aged ready task. That conflated legitimate capacity
    waits with real problems and produced alarmist copy on healthy queues.
    The rework classifies each aged ready task into one of five outcomes
    (Product contract §3) using read-only board/dispatcher evidence:

      LEGITIMATELY_QUEUED / READY_TOO_LONG_UNEXPLAINED /
      NO_COMPATIBLE_WORKER / DISPATCHER_UNHEALTHY /
      PROFILE_CAPACITY_SATURATED

    Fail-safe: when no positive legitimacy evidence exists (low-level calls
    without board context), the outcome defaults to the conservative
    READY_TOO_LONG_UNEXPLAINED warning — a missing signal is never dropped
    for lack of context.
    """
    threshold_seconds = float(
        cfg.get("stranded_threshold_seconds", 30 * 60)
    )
    status = _task_field(task, "status")
    if status != "ready":
        return []
    # Skip tasks with a live claim — they're being worked on, even if
    # the worker hasn't reported progress yet (run-level liveness
    # extends the claim TTL; we don't want to second-guess that here).
    if _task_field(task, "claim_lock"):
        return []
    assignee = _task_field(task, "assignee") or ""
    if not assignee.strip():
        # Unassigned tasks: the dispatcher's ``skipped_unassigned`` is
        # already the right signal. A separate diagnostic here would
        # double-flag the same condition.
        return []

    # Precedence, anti-double-flag (Product §3.3): when repeated failures or
    # crashes already explain this task, the stranded rule cedes the floor.
    failure_threshold = _positive_int(cfg.get("failure_threshold"), 3)
    failures = (
        _task_field(task, "consecutive_failures", None)
        if _task_field(task, "consecutive_failures", None) is not None
        else _task_field(task, "spawn_failures", 0)
    )
    if failures is not None and failures >= failure_threshold:
        return []
    crash_threshold = int(cfg.get("crash_threshold", 2))
    ordered_runs = sorted(runs, key=lambda r: _task_field(r, "id", 0) or 0)
    trailing_crashes = 0
    for r in reversed(ordered_runs):
        outcome = _task_field(r, "outcome")
        if outcome == "crashed":
            trailing_crashes += 1
        elif outcome in ("completed", "reclaimed"):
            break
    if trailing_crashes >= crash_threshold:
        return []

    # Find the most recent event that put this task into ready.
    # ``created`` covers tasks born ready; ``promoted`` covers parent-
    # done auto-promotion; ``reclaimed`` covers TTL/crash recovery;
    # ``unblocked`` covers human-driven resumes.
    READY_TRANSITION_KINDS = {
        "created", "promoted", "reclaimed", "unblocked",
    }
    last_ready_ts = 0
    for ev in events:
        if _event_kind(ev) in READY_TRANSITION_KINDS:
            t = _event_ts(ev)
            last_ready_ts = max(last_ready_ts, t)

    # Fallback: if no qualifying event exists (very old task or events
    # truncated), fall back to ``created_at`` on the task row. Better
    # to occasionally over-flag an ancient task than miss a stranded one.
    if last_ready_ts == 0:
        last_ready_ts = int(_task_field(task, "created_at", default=0) or 0)
    if last_ready_ts == 0:
        return []

    age_seconds = now - last_ready_ts
    if age_seconds < threshold_seconds:
        return []

    board_context = cfg.get("_board_context")
    evidence = _stranded_evidence(task, board_context, now)
    outcome = _classify_ready_outcome(evidence, age_seconds, threshold_seconds)

    classification = outcome["classification"]
    attention = outcome["attention"]
    actions: list[DiagnosticAction] = []
    if classification == CLASSIFICATION_DISPATCHER_UNHEALTHY:
        actions.append(_run_diagnostics_action(suggested=True))
        actions.append(_secondary_cli_hint(
            "hermes kanban diagnostics",
            label="Voir la commande CLI",
        ))
    elif classification == CLASSIFICATION_NO_COMPATIBLE_WORKER:
        actions.append(DiagnosticAction(
            kind="reassign",
            label="Réassigner",
            payload={"current_assignee": assignee},
            suggested=True,
        ))
        actions.append(_run_diagnostics_action())
    elif classification == CLASSIFICATION_READY_TOO_LONG_UNEXPLAINED:
        actions.append(_run_diagnostics_action(suggested=True))
        actions.append(DiagnosticAction(
            kind="reassign",
            label="Réassigner",
            payload={"current_assignee": assignee},
        ))
        actions.append(_secondary_cli_hint(
            "hermes kanban diagnostics",
            label="Voir la commande CLI",
        ))
    elif classification == CLASSIFICATION_PROFILE_CAPACITY_SATURATED:
        if attention == ATTENTION_WARNING:
            actions.append(_run_diagnostics_action(suggested=True))
            actions.append(_view_worker_action())
        else:
            actions.append(_view_queue_action(assignee))
    else:  # LEGITIMATELY_QUEUED
        actions.append(_view_queue_action(assignee))
        actions.append(_view_worker_action())

    task_id = str(_task_field(task, "id") or "")
    detail = (
        f"Task {task_id} has been ready for {_fmt_age(age_seconds)} "
        f"(threshold {int(threshold_seconds)}s). Classification "
        f"{classification}: ready_since={last_ready_ts}, "
        f"assignee={assignee!r}, age_seconds={int(age_seconds)}."
    )
    data = {
        "ready_since": last_ready_ts,
        "age_seconds": int(age_seconds),
        "assignee": assignee,
        "threshold_seconds": int(threshold_seconds),
        "classification": classification,
    }
    data.update(outcome.get("data_extra") or {})

    diag = Diagnostic(
        kind="stranded_in_ready",
        severity=outcome["severity"],
        title=outcome["title"],
        detail=detail,
        actions=actions,
        first_seen_at=last_ready_ts,
        last_seen_at=last_ready_ts,
        count=1,
        data=data,
        **_op(
            attention=attention,
            system_action=outcome["system_action"],
            status=outcome["status"],
            cause=outcome["cause"],
            impact=outcome["impact"],
            classification=classification,
        ),
    )
    diag.attention_banner = outcome["banner"]
    return [diag]


def _rule_recovery_lifecycle(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Surface deterministic auto-recovery lifecycle markers (read-only).

    Bounded and NON-OVERLAPPING with the stall-mission watchdog
    (t_60c940e3 lane): this rule never executes or schedules recovery — it
    renders states that the dispatcher/watchdog record as events
    (``recovery_started`` / ``recovering`` / ``recovery_failed`` /
    ``recovery_succeeded``). Before the merged watchdog emits those markers
    nothing fires (no fabricated RECOVERING).
    """
    state = AUTO_RECOVERY_NONE
    start_ev = None
    fail_ev = None
    for ev in events:
        k = _event_kind(ev)
        if k in _RECOVERY_START_KINDS:
            state = AUTO_RECOVERY_IN_PROGRESS
            start_ev = ev
            fail_ev = None
        elif k == EVENT_RECOVERY_SUCCEEDED:
            state = AUTO_RECOVERY_SUCCEEDED
        elif k == EVENT_RECOVERY_FAILED:
            state = AUTO_RECOVERY_FAILED
            fail_ev = ev

    task_id = str(_task_field(task, "id") or "")
    if state == AUTO_RECOVERY_IN_PROGRESS and start_ev is not None:
        payload = _parse_payload(start_ev)
        action = str(payload.get("action") or payload.get("kind") or "auto-recovery")
        ts = _event_ts(start_ev) or now
        diag = Diagnostic(
            kind="recovery_in_progress",
            severity="warning",
            title="Récupération automatique en cours",
            detail=(
                f"Task {task_id}: deterministic auto-recovery {action!r} is "
                f"in progress (event {_event_kind(start_ev)}). History is kept."
            ),
            actions=[_view_worker_action()],
            first_seen_at=ts,
            last_seen_at=ts,
            count=1,
            data={"action": action, "task_id": task_id},
            **_op(
                attention=ATTENTION_INFO,
                auto_recovery_state=AUTO_RECOVERY_IN_PROGRESS,
                system_action=(
                    f"Hermes exécute {action} de façon sûre ; historique conservé."
                ),
                status=(
                    "Récupération automatique en cours — claim stale / worker "
                    "orphelin détecté."
                ),
                cause=(
                    f"Action {action} déclenchée (claim expiré, pid mort, heartbeat "
                    "absent)."
                ),
                impact="La tâche était bloquée sans worker effectif.",
            ),
        )
        diag.attention_banner = False
        return [diag]
    if state == AUTO_RECOVERY_FAILED and fail_ev is not None:
        payload = _parse_payload(fail_ev)
        action = str(payload.get("action") or payload.get("kind") or "auto-recovery")
        board_impact = bool(payload.get("board_impact"))
        attention = ATTENTION_CRITICAL if board_impact else ATTENTION_ACTION_REQUIRED
        ts = _event_ts(fail_ev) or now
        actions = [
            DiagnosticAction(
                kind="reclaim",
                label="Récupérer la tâche",
                payload={},
                suggested=True,
            ),
            _run_diagnostics_action(),
        ]
        diag = Diagnostic(
            kind="recovery_failed",
            severity="critical" if board_impact else "error",
            title="La récupération automatique a échoué",
            detail=(
                f"Task {task_id}: deterministic auto-recovery {action!r} failed "
                f"(event {_event_kind(fail_ev)}). Escalation to the operator."
            ),
            actions=actions,
            first_seen_at=ts,
            last_seen_at=ts,
            count=1,
            data={
                "action": action,
                "task_id": task_id,
                "board_impact": board_impact,
            },
            **_op(
                attention=attention,
                auto_recovery_state=AUTO_RECOVERY_FAILED,
                system_action="Aucune — la récupération automatique a échoué ; escalade opérateur.",
                status=f"La récupération automatique a échoué : {action}.",
                cause=(
                    "La récupération déterministe n'a pas abouti (échec ou décision "
                    "matérielle requise)."
                ),
                impact=(
                    "Impact board — plusieurs tâches/files peuvent être affectées."
                    if board_impact else
                    "Cette tâche reste bloquée sans worker effectif."
                ),
            ),
        )
        diag.attention_banner = True
        return [diag]
    return []


def _rule_duplicate_implementation(task, events, runs, now, cfg) -> list[Diagnostic]:
    """Two or more active tasks/workers on the same repo+branch+scope.

    Read-only detection driven by board-context evidence (``active_scope_tasks``
    + ``superseded_scope_tasks``). AUTO consolidation only on PROVEN
    supersession; otherwise the owner decides which card continues.
    """
    if _task_field(task, "status") not in ("running", "ready"):
        return []
    board_context = cfg.get("_board_context")
    if not isinstance(board_context, dict):
        return []
    active = board_context.get("active_scope_tasks")
    if not isinstance(active, list) or not active:
        return []
    my_key = _task_scope_key(task)
    if my_key is None:
        return []
    task_id = str(_task_field(task, "id") or "")
    superseded = board_context.get("superseded_scope_tasks") or []
    superseded_me = task_id in superseded
    assignee = str(_task_field(task, "assignee") or "").strip()

    others: list[dict] = []
    for other in active:
        if str(other.get("id") or "") == task_id:
            continue
        if other.get("status") not in ("running", "ready"):
            continue
        if _task_scope_key(other) == my_key:
            others.append(other)
    if not others:
        return []

    other_ids = sorted({str(o.get("id") or "") for o in others})
    ws = str(_task_field(task, "workspace_path") or "")
    branch = str(_task_field(task, "branch_name") or "")
    project = str(_task_field(task, "project_id") or "")
    scope_desc = " + ".join(
        part for part in (ws or None, branch or None, project or None) if part
    )
    attention = (
        ATTENTION_WARNING if superseded_me else ATTENTION_ACTION_REQUIRED
    )
    banner = False if superseded_me else True
    actions = [_view_queue_action(assignee) if assignee else _run_diagnostics_action()]
    if not superseded_me:
        actions.append(DiagnosticAction(
            kind="reassign",
            label="Réassigner",
            payload={"current_assignee": assignee},
        ))
    diag = Diagnostic(
        kind="duplicate_implementation",
        severity="warning",
        title=(
            "Implémentation dupliquée — supersession prouvée (consolidation auto)"
            if superseded_me else
            "Implémentation dupliquée potentielle"
        ),
        detail=(
            f"{len(others)} other active task(s) share the same repo+branch+scope "
            f"({scope_desc or 'unknown'}): {', '.join(other_ids)}."
            + (" This card is proven superseded; auto-consolidation applies."
               if superseded_me else "")
        ),
        actions=actions,
        first_seen_at=now,
        last_seen_at=now,
        count=len(others),
        data={
            "other_task_ids": other_ids,
            "scope": scope_desc,
            "repo": ws,
            "branch": branch,
            "project_id": project,
            "superseded": superseded_me,
        },
        **_op(
            attention=attention,
            system_action=(
                "Consolidation automatique prévue (supersession prouvée)."
                if superseded_me else
                "Aucune consolidation automatique sans supersession prouvée."
            ),
            status=(
                "Implémentation dupliquée — supersession prouvée."
                if superseded_me else
                "Implémentation dupliquée potentielle — même repo+branch+scope."
            ),
            cause=(
                f"{len(others)} tâche(s)/worker(s) actif(s) sur le même périmètre "
                f"({', '.join(other_ids)})."
            ),
            impact="",
            risk="Double implémentation / conflit au merge.",
        ),
    )
    diag.attention_banner = banner
    return [diag]


def _rule_concurrent_writer_risk(task, events, runs, now, cfg) -> list[Diagnostic]:
    """An out-of-band (non-kanban) worker is writing to the same checkout.

    Read-only: evidence arrives via board context (``out_of_band_writers``),
    typically from an observer watching non-kanban processes on the checkout.
    Hermes never touches the checkout; the owner arbitrates.
    """
    if _task_field(task, "status") not in ("running", "ready"):
        return []
    board_context = cfg.get("_board_context")
    if not isinstance(board_context, dict):
        return []
    writers = board_context.get("out_of_band_writers")
    if not isinstance(writers, list) or not writers:
        return []
    checkout = str(_task_field(task, "workspace_path") or "").strip()
    if not checkout:
        return []
    task_id = str(_task_field(task, "id") or "")

    matching = [
        w for w in writers
        if str((w or {}).get("checkout") or "").strip() == checkout
    ]
    if not matching:
        return []
    sources = sorted({str(w.get("source") or "unknown") for w in matching})
    diag = Diagnostic(
        kind="concurrent_writer_risk",
        severity="error",
        title="Worker hors-band détecté sur le même checkout",
        detail=(
            f"Task {task_id}: {len(matching)} out-of-band writer(s) observed on "
            f"checkout {checkout!r} ({', '.join(sources)})."
        ),
        actions=[
            DiagnosticAction(
                kind="comment",
                label="Ajouter un commentaire (arbitrage)",
                payload={},
                suggested=True,
            ),
            _run_diagnostics_action(),
        ],
        first_seen_at=now,
        last_seen_at=now,
        count=len(matching),
        data={
            "checkout": checkout,
            "sources": sources,
            "task_id": task_id,
        },
        **_op(
            attention=ATTENTION_ACTION_REQUIRED,
            system_action="Hermes surveille le checkout ; ne modifie rien.",
            status="Worker hors-band détecté sur le même checkout.",
            cause=(
                "Un worker non-kanban (out-of-band) écrit sur ce checkout."
            ),
            impact="",
            risk="Écritures concurrentes → perte/écrasement.",
        ),
    )
    diag.attention_banner = True
    return [diag]


# Registry — order matters: rules higher on the list render first when
# severity ties. Add new rules here.
_RULES: list[RuleFn] = [
    _rule_hallucinated_cards,
    _rule_triage_aux_unavailable,
    _rule_prose_phantom_refs,
    _rule_repeated_failures,
    _rule_repeated_crashes,
    _rule_review_dependency_deadlock,
    _rule_stuck_in_blocked,
    _rule_block_unblock_cycling,
    _rule_stranded_in_ready,
    _rule_recovery_lifecycle,
    _rule_duplicate_implementation,
    _rule_concurrent_writer_risk,
]


# Known kinds (for the UI's filter / legend / i18n keys). Update when
# rules are added.
DIAGNOSTIC_KINDS = (
    "hallucinated_cards",
    "triage_aux_unavailable",
    "prose_phantom_refs",
    "repeated_failures",
    "repeated_crashes",
    "review_dependency_deadlock",
    "stuck_in_blocked",
    "block_unblock_cycling",
    "stranded_in_ready",
    "recovery_in_progress",
    "recovery_failed",
    "duplicate_implementation",
    "concurrent_writer_risk",
)


# Operator-axis defaults per kind (audit matrix, Product contract §4). Rules
# that need outcome-specific values set them explicitly at construction time;
# the finalizer applies these for the legacy rules that predate the axis.
_KIND_OPERATOR_META: dict[str, dict] = {
    "hallucinated_cards": {
        "attention": ATTENTION_ACTION_REQUIRED,
        "abnormal": True,
        "auto_recoverable": False,
        "system_action": "Aucune — le kernel ne devine pas la route des created_cards fantômes.",
        "status": "Terminaison bloquée : created_cards fantômes déclarés par le worker.",
        "cause": "Le worker a déclaré des id qui n'existent pas ou n'ont pas été créés par son profile.",
        "impact": "La tâche reste dans son état précédent ; les dépendants ne sont pas libérés.",
    },
    "triage_aux_unavailable": {
        "attention": ATTENTION_ACTION_REQUIRED,
        "abnormal": True,
        "auto_recoverable": False,
        "system_action": "Aucune — le dispatcher ne peut pas spécifier/décomposer sans modèle auxiliaire.",
        "status": "Triage bloqué : aucun modèle auxiliaire utilisable.",
        "cause": "Le slot auxiliaire requis n'est pas configuré et aucun modèle principal ne sert de fallback.",
        "impact": "La tâche ne peut pas quitter triage.",
    },
    "prose_phantom_refs": {
        "attention": ATTENTION_INFO,
        "abnormal": False,
        "auto_recoverable": True,
        "system_action": "Auto-clear à la prochaine completion propre.",
        "status": "Terminaison OK ; le résumé référence des task ids inconnus.",
        "cause": "La completion mentionne des id qui ne résolvent pas dans la base du board.",
        "impact": "Aucun — diagnostic informatif.",
    },
    "repeated_failures": {
        "attention": ATTENTION_ACTION_REQUIRED,
        "abnormal": True,
        "auto_recoverable": False,
        "system_action": "Le dispatcher retente puis bloque automatiquement après le failure limit.",
        "status": "Échecs répétés — la tâche ne parvient pas à s'exécuter.",
        "cause": "Chaque tentative échoue de la même façon (spawn/timeout/crash).",
        "impact": "La tâche ne progresse pas ; le circuit breaker finira par la bloquer.",
    },
    "repeated_crashes": {
        "attention": ATTENTION_ACTION_REQUIRED,
        "abnormal": True,
        "auto_recoverable": False,
        "system_action": "Le dispatcher retente jusqu'au circuit breaker.",
        "status": "Le worker crashe de façon répétée.",
        "cause": "Les dernières runs se terminent par outcome=crashed.",
        "impact": "La tâche ne progresse pas sans correction de la cause racine.",
    },
    "review_dependency_deadlock": {
        "attention": ATTENTION_ACTION_REQUIRED,
        "abnormal": True,
        "auto_recoverable": False,
        "system_action": "Aucune — graphe de dépendances et sticky block préservés par conception.",
        "status": "Workflow bloqué : le sticky-block review-required retient des dépendants.",
        "cause": "Un bloc review-required legacy garde des enfants en todo.",
        "impact": "La lane aval ne peut pas démarrer.",
    },
    "stuck_in_blocked": {
        "attention": ATTENTION_ACTION_REQUIRED,
        "abnormal": True,
        "auto_recoverable": False,
        "system_action": "Aucune — une tâche bloquée attend une décision humaine.",
        "status": "Tâche bloquée sans échange récent.",
        "cause": "Aucun commentaire ni tentative d'unblock depuis le passage en blocked.",
        "impact": "La suite du workflow est suspendue.",
    },
    "block_unblock_cycling": {
        "attention": ATTENTION_ACTION_REQUIRED,
        "abnormal": True,
        "auto_recoverable": False,
        "system_action": "Aucune — l'unblock seul ne traite pas la cause racine.",
        "status": "Cycles block→unblock répétés sur cette tâche.",
        "cause": "La tâche est re-bloquée pour sensiblement la même raison après chaque unblock.",
        "impact": "Le workflow oscille sans avancer ; risque de boucle.",
    },
}


DEFAULT_CONFIG = {
    # Match the dispatcher default (kanban.failure_limit) so repeated-failure
    # diagnostics do not lag behind the default auto-block threshold.
    "failure_threshold": 2,
    # Legacy alias accepted at read time by _rule_repeated_failures.
    "spawn_failure_threshold": 2,
    "crash_threshold": 2,
    "blocked_stale_hours": 24,
    # Stranded-task threshold. 30 min by default — below that, the
    # signal is dominated by tasks that are about to be claimed on the
    # next dispatcher tick (default 60s) and would just be noise.
    "stranded_threshold_seconds": 30 * 60,
}


def config_from_kanban_config(kanban_cfg: Optional[dict]) -> dict:
    """Build diagnostics config from the runtime ``kanban`` config section.

    ``kanban.diagnostics.failure_threshold`` remains an explicit override.
    Otherwise, derive the repeated-failure threshold from
    ``kanban.failure_limit`` so CLI/dashboard diagnostics match the
    dispatcher's actual circuit-breaker threshold.
    """
    kanban_cfg = kanban_cfg or {}
    diag_cfg = dict(kanban_cfg.get("diagnostics") or {})
    diag_cfg.setdefault(
        "failure_limit",
        kanban_cfg.get("failure_limit", DEFAULT_CONFIG["failure_threshold"]),
    )
    if (
        "failure_threshold" not in diag_cfg
        and "spawn_failure_threshold" not in diag_cfg
    ):
        diag_cfg["failure_threshold"] = diag_cfg["failure_limit"]
    return diag_cfg


def config_from_runtime_config(raw_config: Optional[dict]) -> dict:
    """Build diagnostics config from the full Hermes runtime config.

    Carries through ``kanban``, ``auxiliary``, and ``model`` keys so triage-
    aware rules can inspect the active aux-helper and main-model state.
    Folds the ``kanban`` block through ``config_from_kanban_config`` so the
    repeated-failure threshold derivation still applies.
    """
    raw_config = raw_config or {}
    if not isinstance(raw_config, dict):
        return {}
    cfg: dict = {}
    kanban_cfg = raw_config.get("kanban")
    if isinstance(kanban_cfg, dict):
        cfg.update(config_from_kanban_config(kanban_cfg))
        cfg["kanban"] = kanban_cfg
    for key in ("auxiliary", "model"):
        value = raw_config.get(key)
        if value is not None:
            cfg[key] = value
    return cfg


def _finalize_operator_fields(diag: Diagnostic) -> None:
    """Fill/validate the operator-attention axis on every emitted diagnostic.

    * ``attention`` — per-kind matrix default when the rule left the default
      INFO (legacy rules); explicit rule values win.
    * ``owner_action`` — always derived from attention (REQUIRED only for
      ACTION_REQUIRED/CRITICAL).
    * ``attention_banner`` — resolved by :func:`attention_banner_policy` when
      the rule did not pin it.
    * FR operator message fallbacks from the matrix for legacy kinds.
    """
    meta = _KIND_OPERATOR_META.get(diag.kind, {})
    if diag.attention == ATTENTION_INFO and meta.get("attention", ATTENTION_INFO) != ATTENTION_INFO:
        diag.attention = meta["attention"]
    # Escalation: unified failure/crash rules that hit CRITICAL technical
    # severity also demand CRITICAL operator attention.
    if (
        diag.kind in ("repeated_failures", "repeated_crashes")
        and diag.severity == "critical"
        and diag.attention == ATTENTION_ACTION_REQUIRED
    ):
        diag.attention = ATTENTION_CRITICAL
    if diag.attention not in ATTENTION_ORDER:
        diag.attention = ATTENTION_INFO
    diag.owner_action = owner_action_for_attention(diag.attention)
    if diag.auto_recovery_state not in AUTO_RECOVERY_STATES:
        diag.auto_recovery_state = AUTO_RECOVERY_NONE
    if diag.attention_banner is None:
        diag.attention_banner = attention_banner_policy(
            attention=diag.attention,
            auto_recovery_state=diag.auto_recovery_state,
            abnormal=bool(meta.get("abnormal", True)),
            auto_recoverable=bool(meta.get("auto_recoverable", False)),
        )
    if not diag.system_action:
        diag.system_action = meta.get("system_action", "")
    if not diag.operator_status:
        diag.operator_status = meta.get("status", "")
    if not diag.operator_cause:
        diag.operator_cause = meta.get("cause", "")
    if not diag.operator_impact:
        diag.operator_impact = meta.get("impact", "")


def compute_task_diagnostics(
    task,
    events: list,
    runs: list,
    *,
    now: Optional[int] = None,
    config: Optional[dict] = None,
    graph: Optional[dict] = None,
    board_context: Optional[dict] = None,
) -> list[Diagnostic]:
    """Run every rule against a single task's state and return a
    severity-sorted list of active diagnostics.

    ``board_context`` carries the read-only board/dispatcher evidence the
    classifier needs (dispatcher health, profile roster/cap, running
    siblings, queue progression, concurrency). Every key is optional;
    absent evidence never counts as health (fail-safe). Build it with
    :func:`build_board_context` or pass it by hand (tests / Mission Control).

    Sorting: critical first, then error, then warning; ties broken by
    most-recent ``last_seen_at``.
    """
    now_ts = int(now if now is not None else time.time())
    config = config or {}
    cfg = {**DEFAULT_CONFIG, **config}
    if graph is not None:
        cfg["_graph"] = graph
    if board_context is not None:
        cfg["_board_context"] = board_context
    if (
        "failure_threshold" not in config
        and "spawn_failure_threshold" not in config
        and "failure_limit" in config
    ):
        cfg["failure_threshold"] = _positive_int(
            config.get("failure_limit"),
            DEFAULT_CONFIG["failure_threshold"],
        )
    out: list[Diagnostic] = []
    for rule in _RULES:
        try:
            out.extend(rule(task, events, runs, now_ts, cfg))
        except Exception:
            # A broken rule must never crash the dashboard. Rule bugs
            # get caught in tests; in production we'd rather drop the
            # diagnostic than 500 a whole /board request.
            continue
    for diag in out:
        _finalize_operator_fields(diag)
    severity_idx = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    attention_idx = {a: i for i, a in enumerate(ATTENTION_ORDER)}
    out.sort(
        key=lambda d: (
            -severity_idx.get(d.severity, -1),
            -attention_idx.get(d.attention, 0),
            -(d.last_seen_at or 0),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Read-only board context collector
#
# Classification decision inputs (Product contract §3) are gathered here from
# STABLE columns at the f1a98f7662 base — no schema change, kanban_db read
# APIs / read-only SELECTs only. Any input the collector cannot observe stays
# None; the classifier treats missing evidence as unknown, never as health.
# ---------------------------------------------------------------------------


def build_board_context(
    conn=None,
    *,
    config: Optional[dict] = None,
    now: Optional[int] = None,
) -> dict:
    """Collect read-only board/dispatcher evidence for the classifier.

    ``conn`` is an optional kanban sqlite3 connection (kanban_db.connect());
    when omitted the collector returns only what config alone can provide
    (profile caps) — safe for low-level callers.

    ``config`` is the diagnostics config dict (see config_from_runtime_config);
    its ``kanban`` subsection supplies ``max_in_progress`` /
    ``max_in_progress_per_profile`` when present.

    Returns the evidence dict consumed by :func:`compute_task_diagnostics`
    via its ``board_context`` argument. Dispatcher health and out-of-band
    writers are intentionally NOT derived here (no reliable read-only source
    at the base): callers with live dispatcher telemetry pass them explicitly.
    """
    now_ts = int(now if now is not None else time.time())
    ctx: dict = {
        "dispatcher": None,
        "profiles": None,
        "lanes": None,
        "profile_cap": None,
        "board_cap": None,
        "running_by_assignee": {},
        "queue_by_assignee": {},
        "queue_progressed_by_assignee": {},
        "expected_slot_freed_by_assignee": {},
        "attempts_by_task": {},
        "active_scope_tasks": [],
        "out_of_band_writers": None,
        "superseded_scope_tasks": [],
    }

    # --- config-derived caps (no DB) ---
    kanban_cfg = config.get("kanban") if isinstance(config, dict) else None
    if isinstance(kanban_cfg, dict):
        raw_per_profile = kanban_cfg.get("max_in_progress_per_profile")
        if isinstance(raw_per_profile, int) and raw_per_profile > 0:
            ctx["profile_cap"] = raw_per_profile
        raw_board = kanban_cfg.get("max_in_progress")
        if isinstance(raw_board, int) and raw_board > 0:
            ctx["board_cap"] = raw_board

    if conn is None:
        return ctx

    try:
        from hermes_cli import kanban_db as kb
    except Exception:
        return ctx

    # --- profile roster (Hermes profiles on disk; positive validity proof) ---
    try:
        profiles = kb.list_profiles_on_disk()
        ctx["profiles"] = [str(p) for p in profiles] if profiles else []
    except Exception:
        ctx["profiles"] = None

    # --- running siblings + ready queue per assignee (stable columns) ---
    heartbeat_max_stale = kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
    claim_ttl = kb.DEFAULT_CLAIM_TTL_SECONDS
    running_map: dict[str, list[dict]] = {}
    queue_map: dict[str, list[dict]] = {}
    try:
        rows = conn.execute(
            "SELECT id, status, assignee, priority, claim_lock, claim_expires, worker_pid, "
            "       last_heartbeat_at, started_at "
            "FROM tasks WHERE status IN ('running', 'ready') AND assignee IS NOT NULL"
        ).fetchall()
        for r in rows:
            assignee = r["assignee"]
            if r["status"] == "running":
                hb = r["last_heartbeat_at"]
                started = r["started_at"]
                stale = False
                if hb:
                    stale = (now_ts - int(hb)) > heartbeat_max_stale
                elif started:
                    stale = (now_ts - int(started)) > claim_ttl
                running_map.setdefault(assignee, []).append({
                    "id": r["id"],
                    "stale": stale,
                    "started_at": r["started_at"],
                    "last_heartbeat_at": hb,
                    "worker_pid": r["worker_pid"],
                })
            else:  # ready — dispatch order is priority DESC then id ASC
                queue_map.setdefault(assignee, []).append({
                    "id": r["id"],
                    "priority": r["priority"] or 0,
                })
    except Exception:
        pass
    ctx["running_by_assignee"] = running_map
    ctx["queue_by_assignee"] = queue_map

    # --- queue progression + freed-slot signals per assignee (event evidence
    # within a bounded window) ---
    window = max(claim_ttl * 2, 2 * 60)
    cutoff = now_ts - window
    progressed_map: dict[str, bool] = {}
    freed_map: dict[str, bool] = {}
    try:
        progressed_rows = conn.execute(
            "SELECT DISTINCT t.assignee AS a FROM task_events e "
            "JOIN tasks t ON t.id = e.task_id "
            "WHERE e.kind IN ('claimed', 'spawned') AND e.created_at >= ? "
            "AND t.assignee IS NOT NULL",
            (cutoff,),
        ).fetchall()
        for r in progressed_rows:
            progressed_map[r["a"]] = True
        freed_rows = conn.execute(
            "SELECT DISTINCT t.assignee AS a FROM task_events e "
            "JOIN tasks t ON t.id = e.task_id "
            "WHERE e.kind IN ('completed', 'reclaimed', 'crashed', 'timed_out') "
            "AND e.created_at >= ? AND t.assignee IS NOT NULL",
            (cutoff,),
        ).fetchall()
        for r in freed_rows:
            freed_map[r["a"]] = True
    except Exception:
        pass
    ctx["queue_progressed_by_assignee"] = progressed_map
    ctx["expected_slot_freed_by_assignee"] = freed_map

    # --- ever-attempted per task (historical runs) ---
    attempts: dict[str, int] = {}
    try:
        for r in conn.execute(
            "SELECT task_id, COUNT(*) AS n FROM task_runs GROUP BY task_id"
        ).fetchall():
            attempts[r["task_id"]] = int(r["n"])
    except Exception:
        pass
    ctx["attempts_by_task"] = attempts

    # --- active same-scope tasks (repo+branch+scope concurrency detection) ---
    try:
        active = conn.execute(
            "SELECT id, status, assignee, workspace_path, branch_name, project_id "
            "FROM tasks WHERE status IN ('running', 'ready')"
        ).fetchall()
        ctx["active_scope_tasks"] = [
            {
                "id": r["id"],
                "status": r["status"],
                "assignee": r["assignee"],
                "workspace_path": r["workspace_path"],
                "branch_name": r["branch_name"],
                "project_id": r["project_id"],
            }
            for r in active
        ]
    except Exception:
        ctx["active_scope_tasks"] = []

    return ctx
